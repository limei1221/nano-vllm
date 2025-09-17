import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def compute_temperature_scaled_probs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        safe_temperatures = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits.div_(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        return probs

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, return_probs: bool = False):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        safe_temperatures = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits.div_(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # epsilon = 1e-10
        # sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  # gumble softmax sampling
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if return_probs:
            return torch.where(temperatures == 0, greedy_tokens, sample_tokens), probs
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
