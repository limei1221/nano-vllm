import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    speculative_model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # speculative_model = None

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=1024,
        enable_chunked_prefill=False,
        speculative_model=speculative_model,
        num_speculative_tokens=5,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # prompts = [
    #     "This is a very long prompt that will definitely need chunked prefill. " * 70 + "introduce yourself"
    # ]  # len(tokens) = 1064
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    token_proposed = 0
    token_accepted = 0
    for prompt, output in zip(prompts, outputs):
        print("\n")
        # print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        token_proposed += output["proposed"]
        token_accepted += output["accepted"]
    accept_rate = output["accepted"] / output["proposed"] if output["proposed"] > 0 else None
    if accept_rate is not None:
        print(f"Accept rate calculated over {len(prompts)} prompts: {accept_rate:.2f}")


if __name__ == "__main__":
    main()
