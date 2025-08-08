import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
import numpy as np
import os


class SpeculativeDecoder:
    """
    Handles speculative decoding using a small model with transformers.
    The small model generates draft tokens that are then verified by the large model.
    """

    def __init__(self, model_path: str, num_speculative_tokens: int = 5):
        self.model_path = model_path
        self.num_speculative_tokens = num_speculative_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate_draft_tokens(
        self,
        sequences: List[List[int]],
        temperatures: List[float]
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Generate draft tokens using the small model.

        Args:
            sequences: List of token sequences
            temperatures: List of temperatures for each sequence

        Returns:
            draft_tokens: List of draft token sequences for each input sequence
            draft_probs: List of probability distributions for each draft token
        """
        draft_tokens = []
        draft_probs = []

        for seq, temp in zip(sequences, temperatures):
            input_ids = torch.tensor([seq], device=self.model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.num_speculative_tokens,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )

            new_tokens = outputs.sequences[0][len(seq):].tolist()
            draft_tokens.append(new_tokens)

            scores_tensor = torch.stack(outputs.scores)  # [num_tokens, 1, vocab_size]
            logits = scores_tensor.squeeze(1)  # [num_tokens, vocab_size]
            if temp > 0:
                logits = logits / temp
            probs = torch.softmax(logits, dim=-1)  # [num_tokens, vocab_size]

            # token_indices = torch.tensor(new_tokens, device=probs.device).unsqueeze(1)  # [num_tokens, 1]
            # token_probs = torch.gather(probs, 1, token_indices).squeeze(1)  # [num_tokens]

            # probs = token_probs.tolist()
            draft_probs.append(probs)

        return draft_tokens, draft_probs


if __name__ == "__main__":
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    num_speculative_tokens = 5
    speculative_decoder = SpeculativeDecoder(
        model_path, num_speculative_tokens)
    prompt = "Introduce yourself."
    prompt = speculative_decoder.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    input_ids = speculative_decoder.tokenizer.encode(prompt)
    sequences = [input_ids]
    temperatures = [1.0]
    draft_tokens, draft_probs = speculative_decoder.generate_draft_tokens(
        sequences, temperatures)
    print('draft_tokens: ', draft_tokens)
    print('draft_probs: ', draft_probs)
    draft_reply = [speculative_decoder.tokenizer.decode(token) for token in draft_tokens][0]
    print('prompt: ', prompt)
    print('draft_reply: ', draft_reply)
