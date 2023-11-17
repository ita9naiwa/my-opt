
class Sequence:
    def __init__(self, prompt_tokens, sample_option):
        assert isinstance(prompt_tokens, list)
        assert isinstance(prompt_tokens[0], int)

        self.prompt_tokens = prompt_tokens
        self.sample_option = sample_option
        self.prompt_kv_filled = False
        self.prompt_kv_indices = None

        self.generated_tokens = []
        self.generated_token_kv_indices = []


    def get_prompt_len(self):
        return len(self.prompt_tokens)

    def init_prompt_kv_cache(self, slots):
        assert self.prompt_kv_indices == None
        self.prompt_kv_indices = slots
        self.prompt_kv_filled = False
