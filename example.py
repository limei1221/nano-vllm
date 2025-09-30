import os
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    path = os.path.expanduser("~/huggingface/Llama-3.2-1B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    if getattr(tokenizer, "chat_template", None):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("--------------------------------\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")

    # # vllm
    # for output in outputs:
    #     print("--------------------------------\n")
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt}")
    #     print(f"Completion: {generated_text}")


if __name__ == "__main__":
    main()
