import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self,
                input_ids, positions,
                k_cache, v_cache,
                offsets,
                cache_indices,
                prompt_init=True,
                new_cache_indices=None):
        pass
