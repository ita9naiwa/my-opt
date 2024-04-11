""" Phi model. Brought from https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_phi.py"""


import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import json
import numpy as np
import os
from os.path import join as pjoin

from transformers import AutoTokenizer

from lm_ops import packed_attention, kv_single_query_attention, rotary_embedding_inplace


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rot_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.dim = rot_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.float16)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)

    def forward(self, q, k, positions, return_cos_sin=False):
        rotary_embedding_inplace(q, k, positions, self.cos_cached, self.sin_cached,
                                 self.head_dim)
        return q, k

class PhiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_scaling =  config["rope_scaling"]
        self.num_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.max_position_embeddings = getattr(config, "n_positions", 2048)
        self.num_key_value_heads = config["num_key_value_heads"]
        self.rope_theta = config["rope_theta"]
        self.partial_rotary_factor = config["partial_rotary_factor"]
        self.qk_layernorm = config["qk_layernorm"]
        self.rotary_dim = int(self.partial_rotary_factor * self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True,
                                dtype=torch.float16)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True,
                                dtype=torch.float16)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True,
                                dtype=torch.float16)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True,
                               dtype=torch.float16)
        rotary_dim = int(self.partial_rotary_factor * self.head_dim)
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"],
                                            elementwise_affine=True,
                                            dtype=torch.float16)
            self.k_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"],
                                            elementwise_affine=True,
                                            dtype=torch.float16)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            rot_dim=self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, h, k_cache, v_cache, positions, offsets, cache_indices,
                prompt_init=True,
                new_cache_indices=None,):
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # if self.qk_layernorm:
        #     q = self.q_layernorm(q)
        #     k = self.k_layernorm(k)
        q, k = self.rotary_emb(q, k, positions)
        if prompt_init:
            k_cache[cache_indices] = k
            v_cache[cache_indices] = v
            s, p, o = packed_attention(q, k, v, offsets, self.num_heads)
        else:
            if new_cache_indices is None:
                raise IndexError("if prompt init is False, "
                                 "new_cache_inidces must be given.")
            # debug
            if q.mean() == torch.nan:
                return
            if k.mean() == torch.nan:
                return
            s, p, o = kv_single_query_attention(q, k, v,
                                                 k_cache, v_cache,
                                                 cache_indices,
                                                 offsets,
                                                 self.num_heads)
            k_cache[new_cache_indices] = k
            v_cache[new_cache_indices] = v
        o = self.dense(o)
        return o

class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PhiMLP(nn.Module):

    def __init__(self,
                 config,):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config["hidden_size"]

        self.fc1 = nn.Linear(config["hidden_size"], n_inner,
                             dtype=torch.float16)
        self.fc2 = nn.Linear(n_inner, config["hidden_size"],
                             dtype=torch.float16)
        self.act = NewGELUActivation()

    def forward(self, h):
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return h

class PhiLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config["hidden_size"],
                                            eps=config["layer_norm_eps"])
        self.attention = PhiAttention(config)
        self.mlp = PhiMLP(config)

    def forward(self, h, k_cache, v_cache, positions, offsets, cache_indices,
                prompt_init=True,
                new_cache_indices=None,):
        residual = h
        h = self.input_layernorm(h)
        attn_output = self.attention(h, k_cache, v_cache, positions, offsets, cache_indices,
                                     prompt_init=prompt_init,
                                     new_cache_indices=new_cache_indices)
        feed_fowrad_output = self.mlp(h)
        h = attn_output + feed_fowrad_output + residual
        return h

class PhiDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList([
            PhiLayer(config) for _ in range(config["num_hidden_layers"])
        ])
        self.final_layer_norm = nn.LayerNorm(config["hidden_size"],
                                             eps=config["layer_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=True,
                                 dtype=torch.float16)
    def forward(self,
                input_ids, positions,
                k_cache, v_cache,
                offsets,
                cache_indices,
                prompt_init=True,
                new_cache_indices=None):
        hs = []
        h = self.embed_tokens(input_ids)
        for i in range(self.config["num_hidden_layers"]):
            hs.append(h.cpu())
            h = self.layers[i](h, k_cache[i], v_cache[i], positions, offsets, cache_indices,
                               prompt_init=prompt_init,
                               new_cache_indices=new_cache_indices)
        h = self.final_layer_norm(h)
        hs.append(h.cpu())
        if prompt_init:
            h = h[offsets - 1]
        ret = self.lm_head(h)
        return ret

    def load_weights(self, model_path):
        model_dict = torch.load(pjoin(model_path, "pytorch_model.bin"))
        model_config = json.load(open(pjoin(model_path, "config.json"), 'r'))

        self.embed_tokens.weight.data = model_dict["model.embed_tokens.weight"]

        self.final_layer_norm.weight.data = model_dict["model.final_layernorm.weight"]
        self.final_layer_norm.bias.data = model_dict["model.final_layernorm.bias"]

        self.lm_head.weight.data = model_dict["lm_head.weight"]
        self.lm_head.bias.data = model_dict["lm_head.bias"]
        not_layers = [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "lm_head.bias",
            "model.final_layernorm.weight",
            "model.final_layernorm.bias"
        ]
        for name, weight in model_dict.items():
            if name in not_layers:
                continue
            # print(name)
            # e.g., "model.layers.0.self_attn.q_proj.weight"
            splitted = name.split(".")
            # ["model", "layers", "0", "self_attn", "q_proj", "weight"]
            # [0         1         2    3            4         5]
            layer_num = int(splitted[2])
            layer = self.layers[layer_num]
            if splitted[3] == "self_attn":
                if splitted[4] == "q_proj":
                    target = layer.attention.q_proj
                elif splitted[4] == "k_proj":
                    target = layer.attention.k_proj
                elif splitted[4] == "v_proj":
                    target = layer.attention.v_proj
                elif splitted[4] == "dense":
                    target = layer.attention.dense
                if splitted[5] == "weight":
                    target.weight.data = weight
                elif splitted[5] == "bias":
                    target.bias.data = weight
            elif splitted[3] == "mlp":
                if splitted[4] == "fc1":
                    target = layer.mlp.fc1
                elif splitted[4] == "fc2":
                    target = layer.mlp.fc2
                if splitted[5] == "weight":
                    target.weight.data = weight
                elif splitted[5] == "bias":
                    target.bias.data = weight
            elif splitted[3] == "input_layernorm":
                if splitted[4] == "weight":
                    layer.input_layernorm.weight.data = weight
                elif splitted[4] == "bias":
                    layer.input_layernorm.bias.data = weight
            elif splitted[3] == "final_layer_norm":
                if splitted[4] == "weight":
                    self.final_layer_norm.weight.data = weight
                elif splitted[4] == "bias":
                    self.final_layer_norm.bias.data = weight
