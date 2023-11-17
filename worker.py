import json
from os.path import join as pjoin

from transformers import AutoTokenizer
import torch

from opt import OPTDecoder
from cache_manager import CacheManager
from sequence import Sequence


class Worker():
    def __init__(self, model_path):
        # load model
        with open(pjoin(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        self.model = OPTDecoder(config)
        self.model.load_weights(pjoin(model_path, "pytorch_model.bin"))
        self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Cache mananger
        self.cache_manager = CacheManager(
            config["num_hidden_layers"],
            config["num_attention_heads"],
            config["hidden_size"],
            1000)

        self.sequence_map = {}
        self.max_concurrent_query = 16
        self.waiting = []
        self.running = []

    def query(self, input):
        prompt, option = input["prompt"], input["args"]
        prompt_tokens = self.tokenizer.encode(prompt)
        query_id = 0
        for i in range(self.max_concurrent_query):
            if i not in self.sequence_map:
                query_id = i
                break
        if i == self.max_concurrent_query:
            raise IndexError("Max concurrent query reached")
        seq = Sequence(prompt_tokens, option)
        prompt_len = seq.get_prompt_len()
        slots = self.cache_manager.get_slots(prompt_len)
        seq.init_prompt_kv_cache(slots)
        self.sequence_map[query_id] = seq
        print(seq.prompt_kv_indices)


if __name__ == "__main__":
    worker = Worker("opt-125m")
    input = {
        "prompt": "I am a gay.",
        "args": {
        }
    }
    worker.query(input)
