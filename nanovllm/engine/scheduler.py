from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_chunked_prefill:
            return self._chunked_prefill_schedule()
        else:
            return self._default_schedule()

    def _default_schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            seq.num_tokens_to_process = len(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _chunked_prefill_schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        token_budget = self.max_num_batched_tokens

        num_seqs = 0
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if token_budget >= 1:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
                    token_budget -= 1
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

        # prefill
        temp_waiting = deque()
        while self.waiting and token_budget > 0 and num_seqs < self.max_num_seqs:
            seq = self.waiting.popleft()

            if (not seq.block_table) and (not self.block_manager.can_allocate(seq)):
                temp_waiting.append(seq)
                continue

            prompt_tokens_left = len(seq) - seq.num_cached_tokens

            if prompt_tokens_left <= token_budget:
                if not seq.block_table:
                    self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                scheduled_seqs.append(seq)
                num_seqs += 1
                seq.num_tokens_to_process = prompt_tokens_left
                token_budget -= prompt_tokens_left
                self.running.append(seq)
            else:
                # Chunk the prompt
                chunk_size = token_budget

                if not seq.block_table:
                    self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                scheduled_seqs.append(seq)
                num_seqs += 1
                seq.num_tokens_to_process = chunk_size
                token_budget = 0

        if temp_waiting:
            self.waiting.extendleft(reversed(temp_waiting))

        if scheduled_seqs:
            return scheduled_seqs, True

        return self._default_schedule()

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # Handle chunked prefill: update cached tokens count
            if seq.num_tokens_to_process is not None:
                seq.num_cached_tokens += seq.num_tokens_to_process
                seq.num_tokens_to_process = None  # Reset for next iteration
                # If we've processed all prompt tokens, move to decode phase
                if seq.num_cached_tokens >= seq.num_prompt_tokens:
                    seq.status = SequenceStatus.RUNNING
                    if seq in self.waiting:
                        self.waiting.remove(seq)
                    if seq not in self.running:
                        self.running.append(seq)

                    seq.append_token(token_id)
                else:
                    # Still in prefill phase, keep in waiting for next chunk
                    seq.status = SequenceStatus.WAITING
                    if seq in self.running:
                        self.running.remove(seq)
                    if seq not in self.waiting:
                        self.waiting.append(seq)
            else:
                # Normal decode phase
                seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
