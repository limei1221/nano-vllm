import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    tokenizer = AutoTokenizer.from_pretrained(model)
    speculative_model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # speculative_model = None

    # llm = LLM(model, enforce_eager=True, tensor_parallel_size=1, enable_chunked_prefill=True, max_model_len=512, max_num_batched_tokens=1024)
    llm = LLM(
        model,
        tensor_parallel_size=1,
        speculative_model=speculative_model,
        num_speculative_tokens=5,
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
    prompts = [
        "introduce yourself",
    ]
    # prompts = [
    #     "This is a very long prompt that will definitely need chunked prefill. " * 70 + "Please provide a comprehensive analysis of artificial intelligence." for _ in range(3)
    # ]
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

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        print(f"Accept rate: {output['accept_rate']:.2f}")


if __name__ == "__main__":
    main()
