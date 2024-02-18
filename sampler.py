import torch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Sampler():
    # t denotes temperature
    def __init__(self, k=-1, p=1.0, t=1.0):
        self.k = k
        self.p = p
        self.t = t
        self.filter_value = -10000.0
    def sample(self, logits, k=None, p=None, t=None):
        ndim = logits.dim()
        if ndim == 1:
            logits = logits.unsqueeze(0)
        elif ndim != 2:
            raise ValueError("logits should be 1D or 2D tensor")

        if k is None:
            k = self.k
        if p is None:
            p = self.p
        if t is None:
            t = self.t

        k = min(k, logits.size(-1))
        p = max(0.0, min(1.0, p))

        # Apply Temperature
        logits = logits / t

        # Apply Top-k
        if k > 0:
            top_k_val = logits.topk(k, dim=-1)[0][:, -1].unsqueeze(-1)
            logits[logits < top_k_val] = self.filter_value

        if p < 1.0:
            # https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html#top_k_top_p_filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = self.filter_value
        ret = torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)
        if ndim == 1:
            ret = ret.squeeze(0)
        return ret

if __name__ == "__main__":
    batch_size = 3
    num_samples = 10
    sampler = Sampler()
    print("non-batch case")
    logits = torch.normal(mean=0, std=1.0, size=(num_samples,))
    sample = sampler.sample(logits, k=30, p=0.7)
    print("sampled", sample)

    print("batched case")
    logits = torch.normal(mean=0, std=1.0, size=(batch_size, num_samples))
    sample = sampler.sample(logits, k=30, p=0.7)
    print("sampled: ", sample)
