import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer, AutoConfig

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.sampling_params import SamplingParams

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # use eager mode instead of cudagraphs
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        self.speculative_model = None
        self.num_speculative_tokens = 0
        self.speculative_decoding = config.speculative_model and config.num_speculative_tokens > 0
        if self.speculative_decoding:
            self.speculative_model_hf_config = AutoConfig.from_pretrained(config.speculative_model)
            self.speculative_model = Qwen3ForCausalLM(self.speculative_model_hf_config)
            load_model(self.speculative_model, config.speculative_model)
            self.num_speculative_tokens = config.num_speculative_tokens

            self.speculative_model_tokenizer = AutoTokenizer.from_pretrained(config.speculative_model, use_fast=True)
            assert self.speculative_model_tokenizer.vocab == self.tokenizer.vocab
        self.vocab_size = self.tokenizer.vocab_size

        self.warmup_model()
        self.allocate_kv_cache()  # allocate kv cache for self.model
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.speculative_decoding:
                self.capture_speculative_cudagraph()
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.speculative_decoding:
                del self.speculative_graphs, self.speculative_graph_pool
                del self.verify_graphs, self.verify_graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):  # for rank > 0
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):  # for rank = 0
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Total bytes available for KV cache allocation (respecting utilization)
        available_bytes = int(total * config.gpu_memory_utilization - used - peak + current)

        target_split = 1.0
        if self.speculative_decoding:
            sconf = self.speculative_model_hf_config
            split_ratio = (hf_config.num_hidden_layers * hf_config.num_key_value_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize) / (sconf.num_hidden_layers * sconf.num_key_value_heads * sconf.head_dim * sconf.torch_dtype.itemsize)
            target_split = split_ratio / (1 + split_ratio)
            assert target_split >= 0.5 and target_split < 1.0
            print('target_split: ', target_split)

        # Split between main and speculative models
        main_budget = int(available_bytes * target_split)
        spec_budget = available_bytes - main_budget

        # Main model KV cache
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = main_budget // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
            dtype=hf_config.torch_dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        config.num_draft_kvcache_block = 0
        self.draft_kv_cache = None
        # Speculative model KV cache, if present
        if self.speculative_decoding and spec_budget > 0:
            s_num_kv_heads = sconf.num_key_value_heads // self.world_size
            s_block_bytes = 2 * sconf.num_hidden_layers * self.block_size * s_num_kv_heads * sconf.head_dim * sconf.torch_dtype.itemsize
            config.num_draft_kvcache_block = spec_budget // s_block_bytes
            assert config.num_draft_kvcache_block > 0
            if config.num_draft_kvcache_block > 0:
                self.draft_kv_cache = torch.zeros(
                    2,
                    sconf.num_hidden_layers,
                    config.num_draft_kvcache_block,
                    self.block_size,
                    s_num_kv_heads,
                    sconf.head_dim,
                    dtype=sconf.torch_dtype,
                )
                layer_id = 0
                for module in self.speculative_model.modules():
                    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                        module.k_cache = self.draft_kv_cache[0, layer_id]
                        module.v_cache = self.draft_kv_cache[1, layer_id]
                        layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if seqs[0].is_draft:
            block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]
        else:
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            if seq.is_draft:
                num_tokens_to_process = seq.draft_num_tokens_to_process
                num_processed_tokens = seq.draft_num_processed_tokens
                block_table = seq.draft_block_table
            else:
                num_tokens_to_process = seq.num_tokens_to_process
                num_processed_tokens = seq.num_processed_tokens
                block_table = seq.block_table
            seqlen = len(seq)
            # Handle chunked prefill: only process specified number of tokens
            if num_tokens_to_process is not None:  # Chunked prefill or Speculative verify or generate draft tokens
                input_ids.extend(seq[num_processed_tokens: num_processed_tokens + num_tokens_to_process])
                positions.extend(list(range(num_processed_tokens, num_processed_tokens + num_tokens_to_process)))
                seqlen_q = num_tokens_to_process
                seqlen_k = num_processed_tokens + num_tokens_to_process
                start_pos = num_processed_tokens
                if block_table:
                    for pos in range(start_pos, start_pos + num_tokens_to_process):
                        block_idx = pos // self.block_size
                        offset_in_block = pos % self.block_size
                        block_id = block_table[block_idx]
                        slot_mapping.append(block_id * self.block_size + offset_in_block)
            else:
                input_ids.extend(seq[seq.num_cached_tokens:])
                positions.extend(list(range(seq.num_cached_tokens, seqlen)))
                seqlen_q = seqlen - seq.num_cached_tokens
                seqlen_k = seqlen
                if seq.block_table:
                    for i in range(seq.num_cached_blocks, seq.num_blocks):
                        start = seq.block_table[i] * self.block_size
                        if i != seq.num_blocks - 1:
                            end = start + self.block_size
                        else:
                            end = start + seq.last_block_num_tokens
                        slot_mapping.extend(list(range(start, end)))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # cumulative sequence lengths
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, self.num_speculative_tokens, True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence], is_draft: bool = False):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            if seq.is_draft:
                block_table = seq.draft_block_table
            else:
                block_table = seq.block_table
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, self.num_speculative_tokens, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify_decode(self, seqs: list[Sequence]):
        # Build (B*(K+1)) tokens for decode verify: last normal token + K draft tokens per seq
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        K = self.num_speculative_tokens
        for seq in seqs:
            block_table = seq.block_table
            start_idx = max(0, len(seq) - K - 1)
            end_idx = len(seq) - 1
            input_ids.extend(seq[start_idx: end_idx + 1])
            positions.extend(list(range(start_idx, end_idx + 1)))
            for pos in range(start_idx, end_idx + 1):
                block_idx = pos // self.block_size
                offset_in_block = pos % self.block_size
                block_id = block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset_in_block)
            context_lens.append(len(seq))

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, self.num_speculative_tokens, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            if context.is_speculative:
                K1 = self.num_speculative_tokens + 1
                assert bs % K1 == 0, "Speculative verify expects (batch_size * (K+1)) tokens"
                B = bs // K1
                graph = self.verify_graphs[next(x for x in self.graph_bs if x >= B)]
                graph_vars = self.verify_graph_vars
                for k, v in graph_vars.items():
                    if k != "outputs":
                        v.zero_()
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"][:B] = context.context_lens
                graph_vars["block_tables"][:B, :context.block_tables.size(1)] = context.block_tables
            else:
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
                graph_vars = self.graph_vars
                for k, v in graph_vars.items():
                    if k != "outputs":
                        v.zero_()
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_speculative_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.speculative_model.compute_logits(self.speculative_model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.speculative_graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.speculative_graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.speculative_model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if not is_prefill and self.speculative_decoding and self.rank == 0:
            return self.run_speculative_decode(seqs, temperatures)

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        self.vocab_size = logits.size(-1)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def run_speculative_decode(self, seqs: list[Sequence], temperatures: torch.Tensor) -> list[int]:
        device = self.model.lm_head.weight.device
        dtype = self.model.lm_head.weight.dtype

        draft_tokens, draft_probs = self.generate_draft_tokens(seqs, temperatures, device, dtype)
        final_token_ids = self.verify_draft_tokens(seqs, draft_tokens, draft_probs, temperatures)

        reset_context()
        return final_token_ids

    @torch.inference_mode()
    def generate_draft_tokens(
        self,
        seqs: list[Sequence],
        temps: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ):
        for seq in seqs:
            seq.is_speculative = False
            seq.is_draft = True
            seq.draft_num_tokens_to_process = seq.num_tokens - seq.draft_num_processed_tokens

        draft_tokens = torch.empty((len(seqs), self.num_speculative_tokens), dtype=torch.int64, device=device)
        draft_probs = torch.empty((len(seqs), self.num_speculative_tokens, self.vocab_size), dtype=dtype, device=device)

        for t in range(self.num_speculative_tokens):
            if t == 0:
                input_ids, positions = self.prepare_prefill(seqs)
                last_logits = self.run_speculative_model(input_ids, positions, True)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                last_logits = self.run_speculative_model(input_ids, positions, False)

            next_tokens, probs = self.sampler(last_logits, temps, return_probs=True)

            draft_tokens[:, t] = next_tokens  # (B,)
            draft_probs[:, t] = probs  # (B, V)

            for i, seq in enumerate(seqs):
                token = int(next_tokens[i].item())
                seq.append_token(token)
                seq.draft_num_processed_tokens += seq.draft_num_tokens_to_process
                seq.draft_num_tokens_to_process = 1

        return draft_tokens, draft_probs

    @torch.inference_mode()
    def verify_draft_tokens(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temps: torch.Tensor
    ) -> list[int]:
        for seq in seqs:
            seq.is_speculative = True
            seq.is_draft = False
            seq.num_tokens_to_process = self.num_speculative_tokens + 1

        input_ids, positions = self.prepare_verify_decode(seqs)
        logits = self.run_model(input_ids, positions, False)
        temps_reshaped = temps.repeat_interleave(self.num_speculative_tokens + 1)
        probs = self.sampler.compute_temperature_scaled_probs(logits, temps_reshaped)
        probs = probs.reshape(len(seqs), -1, probs.size(-1))  # (B, K + 1, V)
        target_probs = probs[:, :self.num_speculative_tokens, :]  # (B, K, V)

        # verify draft tokens
        indices = draft_tokens.unsqueeze(-1)  # (B, K, 1)
        draft_token_probs_from_draft = torch.gather(draft_probs, 2, indices).squeeze(-1)  # (B, K)
        draft_token_probs_from_target = torch.gather(target_probs, 2, indices).squeeze(-1)
        accept_ratio = torch.exp(torch.log(draft_token_probs_from_target) - torch.log(draft_token_probs_from_draft))
        accept_probs = torch.min(torch.ones_like(accept_ratio), accept_ratio)

        accepted = (torch.rand_like(accept_probs) < accept_probs)  # (B, K)
        valid_tokens_mask = torch.cumprod(accepted, dim=1).bool()  # (B, K)
        num_accepted = valid_tokens_mask.sum(dim=1)  # (B,)

        if (num_accepted < self.num_speculative_tokens).any():
            rejection_positions = torch.clamp(num_accepted, max=self.num_speculative_tokens - 1)  # (B,)

            batch_indices = torch.arange(len(seqs), device=rejection_positions.device)  # (B,)
            target_dist_at_rejection = target_probs[batch_indices, rejection_positions]  # (B, V)
            draft_dist_at_rejection = draft_probs[batch_indices, rejection_positions]  # (B, V)

            adjusted_probs = torch.clamp(target_dist_at_rejection - draft_dist_at_rejection, min=0)  # (B, V)
            norm_sum = adjusted_probs.sum(dim=-1, keepdim=True)
            adjusted_probs = adjusted_probs / torch.clamp(norm_sum, min=1e-8)

            all_accepted_mask = (num_accepted == self.num_speculative_tokens)  # (B,)
            last_probs = probs[:, -1, :]  # (B, V)
            final_next_probs = torch.where(all_accepted_mask.unsqueeze(-1), last_probs, adjusted_probs)
        else:
            final_next_probs = probs[:, -1, :]

        final_token_ids = self.sampler(final_next_probs, temps)

        for i, seq in enumerate(seqs):
            accepted_count = num_accepted[i].item()
            seq.num_speculative_proposed_total += self.num_speculative_tokens
            seq.num_speculative_accepted_total += accepted_count
            seq.pending_accepted_tokens = draft_tokens[i, :accepted_count].tolist()

        return final_token_ids.tolist()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, self.num_speculative_tokens, False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_speculative_cudagraph(self):
        config = self.config
        hf_config = self.speculative_model_hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.speculative_graphs = {}
        self.speculative_graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, self.num_speculative_tokens, False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.speculative_model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.speculative_graph_pool):
                outputs[:bs] = self.speculative_model(input_ids[:bs], positions[:bs])    # capture
            if self.speculative_graph_pool is None:
                self.speculative_graph_pool = graph.pool()
            self.speculative_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.speculative_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        # Capture graphs for target model in speculative verify decode path (K+1 tokens per seq)
        config = self.config
        hf_config = config.hf_config
        max_B = min(self.config.max_num_seqs, 512)
        K1 = self.num_speculative_tokens + 1
        max_tokens = max_B * K1
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_B, dtype=torch.int32)
        block_tables = torch.zeros(max_B, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)
        self.verify_graphs = {}
        self.verify_graph_pool = None

        for B in reversed(self.graph_bs):
            if B > max_B:
                continue
            graph = torch.cuda.CUDAGraph()
            n = B * K1
            set_context(True, self.num_speculative_tokens, False,
                        slot_mapping=slot_mapping[:n], context_lens=context_lens[:B], block_tables=block_tables[:B])
            outputs[:n] = self.model(input_ids[:n], positions[:n])  # warmup
            with torch.cuda.graph(graph, self.verify_graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])  # capture
            if self.verify_graph_pool is None:
                self.verify_graph_pool = graph.pool()
            self.verify_graphs[B] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
