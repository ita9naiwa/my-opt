# references:
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# https://github.com/vllm-project/vllm/blob/1b290ace4f0c6b74d7536b1acc831e43e9771527/vllm/model_executor/models/opt.py

import json
from typing import List, Optional, Tuple

import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer

from paged_attention import paged_attention_forward, paged_kv_attention_forward

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU()
}

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)

class OPTAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, h, k_cache, v_cache, offsets, cache_indices, prompt_init=True, new_cache_indices=None):
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # prompt init일 때랑 아닐 때랑 offsets, cache_indices의 목적이 다름
        if prompt_init:
            k_cache[cache_indices] = k
            v_cache[cache_indices] = v
            s, p, o = paged_attention_forward(q, k, v, offsets, self.num_heads)
        else:
            if new_cache_indices is None:
                raise IndexError("if prompt init is False, "
                                 "new_cache_inidces must be given.")
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
        o_proj = self.out_proj(o)
        return o_proj

class OPTLayer(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, activation, ffn_dim, bias, do_layer_norm_before):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_attention_heads = num_attention_heads
        self.activation = ACTIVATION_MAP[activation]
        self.do_layer_norm_before = do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.self_attn = OPTAttention(self.embed_dim, num_attention_heads)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim, bias=bias)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim, bias=bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(self, h, k_cache, v_cache, offsets, cache_indices, prompt_init, new_cache_indices=None):
        r = h
        if self.do_layer_norm_before:
            h = self.self_attn_layer_norm(h)
        h = self.self_attn(h, k_cache, v_cache, offsets, cache_indices, prompt_init, new_cache_indices)

        h = r + h
        if not self.do_layer_norm_before: # 350m do layer norm after attention
            h = self.self_attn_layer_norm(h)
        r = h
        if self.do_layer_norm_before:
            h = self.final_layer_norm(h)
        h = self.activation(self.fc1(h))
        h = self.fc2(h)
        h = r + h
        if not self.do_layer_norm_before:
            h = self.final_layer_norm(h)
        return h

class OPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config["pad_token_id"]
        self.eos_idx = config["eos_token_id"]
        self.max_pos_embs = config["max_position_embeddings"]
        self.vocab_size = config["vocab_size"]
        self.word_embed_proj_dim = config["word_embed_proj_dim"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.hidden_size = config["hidden_size"]
        self.embed_tokens = nn.Embedding(self.vocab_size, self.word_embed_proj_dim)
        self.position_tokens = OPTLearnedPositionalEmbedding(self.max_position_embeddings, self.hidden_size)
        self.do_layer_norm_before = config["do_layer_norm_before"]
        self.num_attention_heads = config["num_attention_heads"]
        self.activation = config["activation_function"]
        self.ffn_dim = config["ffn_dim"]
        self.bias = True
        self.num_hidden_layers = config["num_hidden_layers"]

        if self.word_embed_proj_dim != self.hidden_size:
            self.proj_in = nn.Linear(self.word_embed_proj_dim, self.hidden_size, bias=False)
            self.proj_out = nn.Linear(self.hidden_size, self.word_embed_proj_dim, bias=False)
        else:
            self.proj_in = None
            self.proj_out = None

        if self.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                OPTLayer(self.hidden_size, self.num_attention_heads, self.activation, self.ffn_dim, self.bias, self.do_layer_norm_before)
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(self,
                input_ids, positions,
                k_cache, v_cache,
                offsets,
                cache_indices,
                prompt_init=True,
                new_cache_indices=None):
        emb = self.embed_tokens(input_ids)
        pos = self.position_tokens(positions)
        if self.proj_in is not None:
            emb = self.proj_in(emb)
        h = emb + pos

        for i in range(self.num_hidden_layers):
            h = self.layers[i](h, k_cache[i], v_cache[i],
                               offsets, cache_indices, prompt_init, new_cache_indices)

        if self.final_layer_norm is not None:
            h = self.final_layer_norm(h)
        if self.proj_out is not None:
            h = self.proj_out(h)

        if prompt_init:
            h = h[offsets - 1]
        ret = torch.matmul(model.embed_tokens.weight.data, h.T).T
        return ret


    def load_weights(self, model_path):
        not_layers = ['decoder.embed_tokens.weight',
                      'decoder.embed_positions.weight',
                      'decoder.final_layer_norm.weight',
                      'decoder.final_layer_norm.bias',
                      'decoder.project_out.weight',
                      'decoder.project_in.weight',
                      'lm_head.weight'
                      ]
        model_dict = torch.load(pjoin(model_path, "pytorch_model.bin"))
        model_config = json.load(open(pjoin(model_path, "config.json"), 'r'))

        if self.word_embed_proj_dim != self.hidden_size:
            self.proj_in.weight.data = model_dict["decoder.project_in.weight"].cuda()
            self.proj_out.weight.data = model_dict["decoder.project_out.weight"].cuda()

        self.embed_tokens.weight.data = model_dict["decoder.embed_tokens.weight"].cuda()
        self.position_tokens.weight.data = model_dict["decoder.embed_positions.weight"].cuda()
        if self.final_layer_norm is not None:
            self.final_layer_norm.weight.data = model_dict["decoder.final_layer_norm.weight"].cuda()
            self.final_layer_norm.bias.data = model_dict["decoder.final_layer_norm.bias"].cuda()

        for name, weight in model_dict.items():
            if name in not_layers:
                continue
            splitted_names = name.split('.')
            splitted_names = splitted_names[2:]
            layer_num = int(splitted_names[0])
            splitted_names = splitted_names[1:]
            module_name = splitted_names[0]
            splitted_names = splitted_names[1:]
            if module_name == "self_attn":
                proj = splitted_names[0]
                splitted_names = splitted_names[1:]
                wb = splitted_names[0]
                if proj == "q_proj":
                    if wb == "weight":
                        self.layers[layer_num].self_attn.q_proj.weight.data = weight
                    else:
                        self.layers[layer_num].self_attn.q_proj.bias.data = weight
                elif proj == "k_proj":
                    if wb == "weight":
                        self.layers[layer_num].self_attn.k_proj.weight.data = weight
                    else:
                        self.layers[layer_num].self_attn.k_proj.bias.data = weight
                elif proj == "v_proj":
                    if wb == "weight":
                        self.layers[layer_num].self_attn.v_proj.weight.data = weight
                    else:
                        self.layers[layer_num].self_attn.v_proj.bias.data = weight
                elif proj == "out_proj":
                    if wb == "weight":
                        self.layers[layer_num].self_attn.out_proj.weight.data = weight
                    else:
                        self.layers[layer_num].self_attn.out_proj.bias.data = weight
            elif module_name == "self_attn_layer_norm":
                wb = splitted_names[0]
                if wb == "weight":
                    self.layers[layer_num].self_attn_layer_norm.weight.data = weight
                else:
                    self.layers[layer_num].self_attn_layer_norm.bias.data = weight
            elif module_name == "fc1":
                wb = splitted_names[0]
                if wb == "weight":
                    self.layers[layer_num].fc1.weight.data = weight
                else:
                    self.layers[layer_num].fc1.bias.data = weight
            elif module_name == "fc2":
                wb = splitted_names[0]
                if wb == "weight":
                    self.layers[layer_num].fc2.weight.data = weight
                else:
                    self.layers[layer_num].fc2.bias.data = weight
            elif module_name == "final_layer_norm":
                wb = splitted_names[0]
                if wb == "weight":
                    self.layers[layer_num].final_layer_norm.weight.data = weight
                else:
                    self.layers[layer_num].final_layer_norm.bias.data = weight

if __name__ == "__main__":
    from os.path import join as pjoin
    import os
    import time
    from sampler import Sampler
    s = Sampler(k=50, p=0.9, t=1.0)

    model_name = "opt-350m"
    with open(pjoin(model_name, "config.json"), 'r') as f:
        config = json.load(f)
    model = OPTDecoder(config)
    model.load_weights(pjoin(model_name))
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    v_cache = [torch.zeros(8192, config["hidden_size"]).to(torch.float16).cuda() for _ in range(config["num_hidden_layers"])]
    k_cache = [torch.zeros(8192, config["hidden_size"]).to(torch.float16).cuda() for _ in range(config["num_hidden_layers"])]

    sentences = [
        "Seoul is a city",
    ]
    generated_sentences = ["" for _ in range(len(sentences))]
    total_sampled_tokens = [[] for _ in range(len(sentences))]

    token_ids = [tokenizer.encode(sentence) for sentence in sentences]
    cache_indices = [(256 * i + np.arange(len(tokens))).tolist() for (i, tokens) in enumerate(token_ids)]

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
    for i in range(1):
        new_token_ids = sampled_tokens
        def build_generation_input(sampled_tokens, list_of_cache_indices):
            new_positions = [len(x) for x in list_of_cache_indices]
            offsets = np.cumsum(new_positions).tolist()
            cache_indices = np.concatenate(list_of_cache_indices).tolist()
            new_cache_indices = [256 * i + len(cache_indices) for (i, cache_indices) in enumerate(list_of_cache_indices)]
            return (torch.IntTensor(sampled_tokens).cuda(),
                    torch.IntTensor(new_positions).cuda(),
                    torch.IntTensor(offsets).cuda(),
                    torch.IntTensor(cache_indices).cuda(),
                    torch.IntTensor(new_cache_indices).cuda())
        new_token_ids, new_positions, offsets, input_cache_indices, new_cache_indices = build_generation_input(new_token_ids, cache_indices)
        print("new_token_ids", new_token_ids.shape)
        ret = model.forward(new_token_ids,
                            new_positions,
                            k_cache, v_cache,
                            offsets,
                            input_cache_indices,
                            prompt_init=False,
                            new_cache_indices=new_cache_indices)
        sampled_tokens = s.sample(ret).cpu().numpy().tolist()
        print("sampled tokens", sampled_tokens)
        de = tokenizer.batch_decode(sampled_tokens)
        _nc = new_cache_indices.cpu().numpy().tolist()
        break
        os.system("clear")
        for i in range(len(sentences)):
            cache_indices[i].append(_nc[i])
            total_sampled_tokens[i].append(sampled_tokens[i])
            generated_sentences[i] += de[i]
            print("====================================================")
            print("sentence ", i, sentences[i] + generated_sentences[i])
            print("====================================================")
        # time.sleep(0.5)