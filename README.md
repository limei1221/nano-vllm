# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Online serving

- **Start the server**:
```bash
python server.py --model ~/huggingface/Qwen3-1.7B --host 0.0.0.0 --port 8000
```
or
```bash
python server.py --model ~/huggingface/Qwen3-1.7B \
  --speculative-model ~/huggingface/Qwen3-0.6B \
  --num-speculative-tokens 5 \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

- **Send a request**:
```bash
python client.py --base-url http://localhost:8000 --message "Introduce yourself."
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4090 (24GB)
- Model: Qwen3-1.7B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens
- Config: max_model_len=2048, max_num_seqs=128, max_num_batched_tokens=4096

**Performance Results:**
|   | enable_chunked_prefill | speculative_model | num_speculative_tokens | Output Tokens | Time (s) | Throughput (tokens/s) |
|---|---|---|---|---|---|---|
| | False | None       | 0    | 133,966 | 75.18 | 1782.01 |
| | True  | None       | 0    | 133,966 | 80.01 | 1674.44 |
| | False | Qwen3-0.6B | 5    | 133,966 |  |  |
