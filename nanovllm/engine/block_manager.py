from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, num_draft_blocks: int = 0):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        self.speculative_decoding = False
        self.draft_blocks = None
        self.free_draft_block_ids = None
        self.used_draft_block_ids = None
        if num_draft_blocks > 0:
            self.speculative_decoding = True
            self.draft_blocks: list[Block] = [Block(i) for i in range(num_draft_blocks)]
            self.free_draft_block_ids: deque[int] = deque(range(num_draft_blocks))
            self.used_draft_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _allocate_draft_block(self, block_id: int) -> Block:
        block = self.draft_blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_draft_block_ids.remove(block_id)
        self.used_draft_block_ids.add(block_id)
        return self.draft_blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _deallocate_draft_block(self, block_id: int) -> Block:
        assert self.draft_blocks[block_id].ref_count == 0
        self.used_draft_block_ids.remove(block_id)
        self.free_draft_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        if not self.speculative_decoding:
            return len(self.free_block_ids) >= seq.num_blocks
        else:
            return len(self.free_block_ids) >= seq.num_blocks and len(self.free_draft_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # Allocate draft blocks
        if self.speculative_decoding:
            assert not seq.draft_block_table
            for i in range(seq.num_blocks):
                block_id = self.free_draft_block_ids[0]
                block = self._allocate_draft_block(block_id)
                seq.draft_block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        # Deallocate draft blocks
        if self.speculative_decoding:
            for block_id in reversed(seq.draft_block_table):
                block = self.draft_blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_draft_block(block_id)
            seq.draft_block_table.clear()

    def can_append(self, seq: Sequence, speculative_decoding: bool = False, num_speculative_tokens: int = 0) -> bool:
        if speculative_decoding:
            target_len = len(seq) + num_speculative_tokens
            needed_blocks = (target_len + self.block_size - 1) // self.block_size
            current_blocks = len(seq.block_table)
            required_blocks = max(0, needed_blocks - current_blocks)
        else:
            required_blocks = 1 if (len(seq) % self.block_size == 1) else 0

        if speculative_decoding:
            return len(self.free_block_ids) >= required_blocks and len(self.free_draft_block_ids) >= required_blocks
        else:
            return len(self.free_block_ids) >= required_blocks

    def may_append(self, seq: Sequence, speculative_decoding: bool, num_speculative_tokens: int):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if speculative_decoding:
            if len(seq) + num_speculative_tokens > len(block_table) * self.block_size:  # a new block needs to be allocated
                # assert last_block.hash != -1
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                self._allocate_draft_block(block_id)
                block_table.append(block_id)
            if len(seq) % self.block_size <= num_speculative_tokens:  # the last block gets finalized with a hash
                if last_block.hash != -1:
                    token_ids = seq.block(seq.num_blocks-1)
                    prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_block.update(h, token_ids)
                    self.hash_to_block_id[h] = last_block.block_id
            else:
                assert last_block.hash == -1
        else:
            if len(seq) % self.block_size == 1:  # a new block needs to be allocated
                assert last_block.hash != -1
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            elif len(seq) % self.block_size == 0:  # the last block gets finalized with a hash
                assert last_block.hash == -1
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            else:
                assert last_block.hash == -1
