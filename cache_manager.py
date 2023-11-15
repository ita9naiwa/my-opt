import torch

class CacheManager:
    def __init__(self, n_layers, n_heads, embed_dim, num_tokens, dtype=torch.float16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.dtype = dtype
        self.k_cache = None
        self.v_cache = None
        self.empty_slot = []

    def init_cache(self):
        self.v_cache = [
            torch.zeros(self.num_tokens, self.embed_dim).cuda()
            for _ in range(self.n_layers)]
        self.k_cache = [
            torch.zeros(self.num_tokens, self.embed_dim).cuda()
            for _ in range(self.n_layers)]
        self.empty_slot = list(range(self.num_tokens))

    def get_slots(self, size=1):
        if len(self.empty_slot) < size:
            return None
        if size == 1:
            slot = self.empty_slot[0]
            self.empty_slot.pop(0)
            return slot
        elif size > 1:
            slots = self.empty_slot[:size]
            self.empty_slot = self.empty_slot[size:]
            return slots

    def free_slots(self, slot):
        if isinstance(slot, int):
            self.empty_slot.append(slot)
        elif isinstance(slot, list):
            self.empty_slot.extend(slot)
