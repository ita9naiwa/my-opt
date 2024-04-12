import json
import os
import time
from models import opt
from models.sampler import Sampler
from transformers import AutoTokenizer
import torch
import numpy as np
from os.path import join as pjoin

class OPTModel:
    def __init__(self, model_name, cache_size):
        self.model_name = model_name
        self.cache_size = cache_size
        with open(pjoin(model_name, "config.json"), 'r') as f:
            config = json.load(f)
        self.model = opt.OPTDecoder(config)
        self.model.load_weights(model_name)
        self.model = self.model.cuda()
        self.k_cache = [torch.zeros(self.cache_size, config["hidden_size"]).to(torch.float16).cuda()
                        for _ in range(config["num_hidden_layers"])]
        self.v_cache = [torch.zeros(self.cache_size, config["hidden_size"]).to(torch.float16).cuda()
                        for _ in range(config["num_hidden_layers"])]

def new(gamma=5):
    sampler = Sampler(k=50, p=0.9, t=1.0)
    model_q = OPTModel("opt-125m", 1024)
    model_p = OPTModel("opt-350m", 1024)
    tokenizer = AutoTokenizer.from_pretrained("opt-125m") # two tokenizers are same actually

    sentences = ["Seoul is a city"]
    generated_sentences = ["" for _ in range(len(sentences))]
    total_sampled_tokens = [[] for _ in range(len(sentences))]

    token_ids = [tokenizer.encode(sentence) for sentence in sentences]
    cache_indices_q = [(256 * i + np.arange(len(tokens))).tolist() for (i, tokens) in enumerate(token_ids)]
    cache_indices_p = [(256 * i + np.arange(len(tokens))).tolist() for (i, tokens) in enumerate(token_ids)]


    def build_prompt_input(list_of_token_ids, list_of_cache_indices):
        offsets = torch.from_numpy(np.cumsum([len(token_ids) for token_ids in list_of_token_ids])).int()
        token_ids = np.concatenate(list_of_token_ids)
        positions = np.concatenate([np.arange(len(token_ids)) for token_ids in list_of_token_ids])
        cache_indices = np.concatenate(list_of_cache_indices)
        return (torch.IntTensor(offsets).cuda(),
                torch.IntTensor(token_ids).cuda(),
                torch.IntTensor(positions).cuda(),
                torch.IntTensor(cache_indices).cuda())

    print("------init step------")
    offsets, input_token_ids, input_positions, input_cache_indices = build_prompt_input(token_ids, cache_indices_q)
    logit_q = model_q.model.forward(input_token_ids, input_positions, model_q.k_cache, model_q.v_cache, offsets, input_cache_indices, prompt_init=True)

    offsets, input_token_ids, input_positions, input_cache_indices = build_prompt_input(token_ids, cache_indices_p)
    logit_p = model_p.model.forward(input_token_ids, input_positions, model_p.k_cache, model_p.v_cache, offsets, input_cache_indices, prompt_init=True)
    # here just sample from p
    print(logit_p)
    sampled_tokens = sampler.sample(logit_p).cpu().numpy().tolist()
    for i in range(len(sentences)):
        total_sampled_tokens[i].append(sampled_tokens[i])
        generated_sentences[i] += tokenizer.batch_decode(sampled_tokens)[i]
        print("====================================================")
        print("sentence ", i, sentences[i] + generated_sentences[i])
        print("====================================================")

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

    # # handle single token id at the generation stage
    for iter in range(5):
        current_logits_q = []
        current_sampled_tokens = [[sampled_tokens[i]] for _ in range(len(sentences))]
        for i in range(gamma):
            new_token_ids = sampled_tokens
            def build_generation_input_for_q(sampled_tokens, list_of_cache_indices):
                new_positions = [len(x) for x in list_of_cache_indices]
                offsets = np.cumsum(new_positions).tolist()
                cache_indices = np.concatenate(list_of_cache_indices).tolist()
                new_cache_indices = [256 * i + len(cache_indices) for (i, cache_indices) in enumerate(list_of_cache_indices)]
                return (torch.IntTensor(sampled_tokens).cuda(),
                        torch.IntTensor(new_positions).cuda(),
                        torch.IntTensor(offsets).cuda(),
                        torch.IntTensor(cache_indices).cuda(),
                        torch.IntTensor(new_cache_indices).cuda())

            new_token_ids, new_positions, offsets, input_cache_indices, new_cache_indices = build_generation_input_for_q(new_token_ids, cache_indices_q)
            logit_q = model_q.model.forward(new_token_ids,
                                new_positions,
                                model_q.k_cache, model_q.v_cache,
                                offsets,
                                input_cache_indices,
                                prompt_init=False,
                                new_cache_indices=new_cache_indices)
            sampled_tokens = sampler.sample(logit_q).cpu().numpy().tolist()
            de = tokenizer.batch_decode(sampled_tokens)
            _nc = new_cache_indices.cpu().numpy().tolist()
            for i in range(len(sentences)):
                cache_indices_q[i].append(_nc[i])
                current_logits_q.append(logit_q)
                current_sampled_tokens[i].append(sampled_tokens[i])

        logits_q = torch.stack(current_logits_q, dim=1)
        def build_generation_input_for_p(sampled_tokens, list_of_cache_indices):
            new_positions = [len(x) for x in list_of_cache_indices]
            offsets = np.cumsum(new_positions).tolist()
            cache_indices = np.concatenate(list_of_cache_indices).tolist()
            new_cache_indices = [[256 * i + len(cache_indices_p[i]) + j for j in range(gamma + 1)]
                                for i in range(len(list_of_cache_indices))]

            return (torch.IntTensor(sampled_tokens).cuda(),
                    torch.IntTensor(new_positions).cuda(),
                    torch.IntTensor(offsets).cuda(),
                    torch.IntTensor(cache_indices).cuda(),
                    torch.IntTensor(new_cache_indices).cuda())

        new_token_ids, new_positions, offsets, input_cache_indices, new_cache_indices = \
            build_generation_input_for_p(current_sampled_tokens, cache_indices_p)
        logits_p = model_p.model.forward(new_token_ids,
                            new_positions,
                            model_p.k_cache, model_p.v_cache,
                            offsets,
                            input_cache_indices,
                            prompt_init=False,
                            new_cache_indices=new_cache_indices,
                            multi_query_multi_cache=True)


        for i in range(len(sentences)):
            curr_sampled_tokens_i = torch.tensor(current_sampled_tokens[i]).cuda().unsqueeze(0)
            prob_q_i = torch.softmax(logits_q[i], dim=-1)
            prob_p_i = torch.softmax(logits_p[i], dim=-1)
            chosen_prob_q_i = torch.gather(prob_q_i, 1, curr_sampled_tokens_i)
            chosen_prob_p_i = torch.gather(prob_p_i, 1, curr_sampled_tokens_i)
            # print(chosen_prob_p_i)
            # print(chosen_prob_q_i)

            r = torch.rand(chosen_prob_q_i.shape).cuda()
            idx = (r > (chosen_prob_p_i / chosen_prob_q_i)).squeeze(0)
            # tmp = [gamma]
            tmp = [gamma]
            for idx, k in enumerate(idx):
                if k:
                    tmp.append(idx)
            chosen_til = min(tmp)
            cache_indices_p[i].extend([new_cache_indices[i][j].cpu().numpy().tolist() for j in range(chosen_til)])
            cache_indices_q[i] = cache_indices_p[i]
            total_sampled_tokens[i].extend(current_sampled_tokens[i][1:chosen_til + 1])
            generated_sentences[i] += tokenizer.decode(current_sampled_tokens[i][1:chosen_til + 1])


            os.system("clear")
            print("====================================================")
            print("sentence ", i, sentences[i] + generated_sentences[i])
            print("====================================================")



new()