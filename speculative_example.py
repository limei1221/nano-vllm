import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Model paths
    large_model = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    small_model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Initialize tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(large_model)
    
    # Initialize LLM with speculative decoding
    llm = LLM(
        large_model,
        tensor_parallel_size=1,
        speculative_model=small_model,  # Small model for draft generation
        num_speculative_tokens=5,       # Number of draft tokens to generate
        enforce_eager=True,             # Disable CUDA graphs for speculative decoding
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,  # Use temperature > 0 for sampling
        max_tokens=100,
        ignore_eos=False
    )
    
    # Test prompts
    prompts = [
        "introduce yourself",
    ]
    
    # Apply chat template if needed
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for prompt in prompts
    ]
    
    print("Generating with speculative decoding...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Print results
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")
        print(f"Tokens generated: {len(output['token_ids'])}")


if __name__ == "__main__":
    main()
