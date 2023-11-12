# references:
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# https://github.com/vllm-project/vllm/blob/1b290ace4f0c6b74d7536b1acc831e43e9771527/vllm/model_executor/models/opt.py

import json
from typing import List, Optional, Tuple

import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer

from attention import naive_attention_forward, kv_attention_forward

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

    def forward(self, h, k_cache, v_cache):
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        if k_cache is not None:
            _, _, o = kv_attention_forward(q, k, v, k_cache, v_cache, self.num_heads)
        else:
            batch_size = q.size(0)
            context_size = q.size(1)
            mask = torch.from_numpy(np.tril(np.ones(context_size).astype(np.float32))).reshape(1, context_size, context_size).repeat(batch_size, 1, 1).cuda()
            _, _, o = naive_attention_forward(q, k, v, mask, self.num_heads)

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

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn = OPTAttention(self.embed_dim, num_attention_heads)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim, bias=bias)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim, bias=bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, h, k_cache, v_cache):
        r = h
        if self.do_layer_norm_before:
            h = self.self_attn_layer_norm(h)
        h = self.self_attn(h, k_cache, v_cache)
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
            self.final_lyaer_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.final_lyaer_norm = None

        self.layers = nn.ModuleList(
            [
                OPTLayer(self.hidden_size, self.num_attention_heads, self.activation, self.ffn_dim, self.bias, self.do_layer_norm_before)
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(self, input_ids, positions, k_cache, v_cache):
        emb = self.embed_tokens(input_ids)
        pos = self.position_tokens(positions)
        if self.proj_in is not None:
            emb = self.proj_in(emb)
        h = emb + pos

        for i in range(self.num_hidden_layers):
            if k_cache is not None:
                h = self.layers[i](h, k_cache[i], v_cache[i])
            else:
                h = self.layers[i](h, None, None)
        if self.final_lyaer_norm is not None:
            h = self.final_lyaer_norm(h)
        if self.proj_out is not None:
            h = self.proj_out(h)
        return h

    def load_weights(self, model_bin_path):
        not_layers = ['model.decoder.embed_tokens.weight',
         'model.decoder.embed_positions.weight',
         'model.decoder.final_layer_norm.weight',
         'model.decoder.final_layer_norm.bias',
         'lm_head.weight']
        model_dict = torch.load(model_bin_path)
        self.embed_tokens.weight = model_dict["model.decoder.embed_tokens.weight"]
        self.position_tokens.weight = model_dict["model.decoder.embed_positions.weight"]

        if self.final_layer_norm is not None:
            self.final_lyaer_norm.weight = model_dict["model.decoder.final_layer_norm.weight"]
            self.final_layer_norm.bias = model_dict["model.decoder.final_layer_norm.bias"]

        for name, weight in model_bin_path.items():
            if name in not_layers:
                continue
            splitted_names = name.split('.')
            splitted_names = splitted_names[3:]
            layer_num = int(splitted_names[0])
            print(layer_num)

if __name__ == "__main__":
    with open("opt-125m/config.json", 'r') as f:
        config = json.load(f)
    model = OPTDecoder(config).cuda()

    tokenizer = AutoTokenizer.from_pretrained("./opt-125m/")
    token_ids = tokenizer.encode("I am a gay")
    token_ids = torch.LongTensor(token_ids).unsqueeze(0).cuda()
    positions = torch.LongTensor(list(range(len(token_ids)))).unsqueeze(0).cuda()
    with torch.no_grad():
        ret = model.forward(token_ids, positions, None, None)
        ret.shape