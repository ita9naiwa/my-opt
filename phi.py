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

from paged_attention import paged_attention_forward, paged_kv_attention_forward



# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Phi
class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
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
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q, k, positions, return_cos_sin=False):
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if return_cos_sin:
            return cos, sin, q, k
        else:
            return q, k

QQ = []
KK = []
PP = []
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads
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

        self.rotary_emb = PhiRotaryEmbedding(
            self.rotary_dim,
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

        q = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        q_rot, q_pass = (q[..., : self.rotary_dim], q[..., self.rotary_dim :])
        k_rot, k_pass = (k[..., : self.rotary_dim], k[..., self.rotary_dim :])
        q_rot, k_rot = self.rotary_emb(q_rot, k_rot, positions)
        q = torch.cat((q_rot, q_pass), dim=-1).transpose(0, 1)
        k = torch.cat((k_rot, k_pass), dim=-1).transpose(0, 1)
        q = q.reshape(-1, self.hidden_size)
        k = k.reshape(-1, self.hidden_size)
        if prompt_init:
            k_cache[cache_indices] = k
            v_cache[cache_indices] = v
            s, p, o = paged_attention_forward(q, k, v, offsets, self.num_heads)
        else:
            if new_cache_indices is None:
                raise IndexError("if prompt init is False, "
                                 "new_cache_inidces must be given.")
            # debug
            if q.mean() == torch.nan:
                return
            if k.mean() == torch.nan:
                return
            s, p, o = paged_kv_attention_forward(q, k, v,
                                                 k_cache, v_cache,
                                                 cache_indices,
                                                 offsets,
                                                 self.num_heads)
            k_cache[new_cache_indices] = k
            v_cache[new_cache_indices] = v
        QQ.append(q.cpu())
        KK.append(k.cpu())
        PP.append(p.cpu())
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

if __name__ == "__main__":
    CACHE_SIZE = 1024
    MAX_SEQ_SIZE = 256
    model_name = "phi-1_5"
    with open(pjoin(model_name, "config.json"), 'r') as f:
            config = json.load(f)
    from sampler import Sampler
    s = Sampler(k=50, p=0.9, t=1.0)

    model = PhiDecoder(config)
    model.load_weights(model_name)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    v_cache = [torch.zeros(CACHE_SIZE, config["hidden_size"]).to(torch.float16).cuda() for _ in range(config["num_hidden_layers"])]
    k_cache = [torch.zeros(CACHE_SIZE, config["hidden_size"]).to(torch.float16).cuda() for _ in range(config["num_hidden_layers"])]

    sentences = [
    """Let me tell you a story: """,
    """sex is fun """,
    ]
    generated_sentences = ["" for _ in range(len(sentences))]
    total_sampled_tokens = [[] for _ in range(len(sentences))]

    token_ids = [tokenizer.encode(sentence) for sentence in sentences]
    cache_indices = [(MAX_SEQ_SIZE * i + np.arange(len(tokens))).tolist() for (i, tokens) in enumerate(token_ids)]
    print("cache_indices", cache_indices)
    # handle list of token ids at the prompt stage
    def build_prompt_input(list_of_token_ids, list_of_cache_indices):
        offsets = torch.from_numpy(np.cumsum([len(token_ids) for token_ids in list_of_token_ids])).int()
        token_ids = np.concatenate(list_of_token_ids)
        positions = np.concatenate([np.arange(len(token_ids)) for token_ids in list_of_token_ids])
        cache_indices = np.concatenate(list_of_cache_indices)
        return (torch.IntTensor(offsets).cuda(),
                torch.IntTensor(token_ids).cuda(),
                torch.IntTensor(positions).cuda(),
                torch.IntTensor(cache_indices).cuda())

    offsets, input_token_ids, input_positions, input_cache_indices = build_prompt_input(token_ids, cache_indices)

    ret = model.forward(input_token_ids, input_positions, k_cache, v_cache, offsets, input_cache_indices, prompt_init=True)
    sampled_tokens = s.sample(ret).cpu().numpy().tolist()
    de = tokenizer.batch_decode(sampled_tokens)

    for i in range(len(sentences)):
        total_sampled_tokens[i].append(sampled_tokens[i])
        generated_sentences[i] += de[i]
        print("====================================================")
        print("sentence ", i, sentences[i] + generated_sentences[i])
        print("====================================================")

    # # handle single token id at the generation stage
    for i in range(MAX_SEQ_SIZE - 1):
        new_token_ids = sampled_tokens
        def build_generation_input(sampled_tokens, list_of_cache_indices):
            new_positions = [len(x) for x in list_of_cache_indices]
            offsets = np.cumsum(new_positions).tolist()
            cache_indices = np.concatenate(list_of_cache_indices).tolist()
            new_cache_indices = [MAX_SEQ_SIZE * i + len(cache_indices) for (i, cache_indices) in enumerate(list_of_cache_indices)]
            return (torch.IntTensor(sampled_tokens).cuda(),
                    torch.IntTensor(new_positions).cuda(),
                    torch.IntTensor(offsets).cuda(),
                    torch.IntTensor(cache_indices).cuda(),
                    torch.IntTensor(new_cache_indices).cuda())
        new_token_ids, new_positions, offsets, input_cache_indices, new_cache_indices = build_generation_input(new_token_ids, cache_indices)
        ret = model.forward(new_token_ids,
                            new_positions,
                            k_cache, v_cache,
                            offsets,
                            input_cache_indices,
                            prompt_init=False,
                            new_cache_indices=new_cache_indices)
        sampled_tokens = s.sample(ret).cpu().numpy().tolist()
        de = tokenizer.batch_decode(sampled_tokens)
        _nc = new_cache_indices.cpu().numpy().tolist()
        os.system("clear")
        for i in range(len(sentences)):
            cache_indices[i].append(_nc[i])
            total_sampled_tokens[i].append(sampled_tokens[i])
            generated_sentences[i] += de[i]
            print("====================================================")
            print("sentence ", i, sentences[i] + generated_sentences[i])
            print("====================================================")

    #     # time.sleep(0.5)