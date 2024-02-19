import os

from models.phi import PhiDecoder
from os.path import join as pjoin
from transformers import AutoTokenizer

import torch
from torch import nn
import json
import numpy as np



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