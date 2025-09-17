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
- Hardware: RTX 4090 (48GB)
- Model: Qwen3-1.7B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens (default)
- Output Length: Randomly sampled between 100–1024 tokens
- max_model_len=2048 (default)
- max_num_batched_tokens=4096 (default)
- max_num_seqs=128

**Performance results:**
temperature=0.6
| | enable_chunked_prefill | input_length | max_model_len | max_num_batched_tokens | Output Tokens | Time (s) | Throughput (tokens/s) |
|---|---|---|---|---|---|---|---|
| | False | 100-1024 | 2048 | 4096   | 133,966 | 28.87 | 4640.72 |
| | True  | 100-1024 | 2048 | 4096   | 133,966 | 27.95 | 4793.67 |
| | False | 1025-1280 | 512 | 1024   | - | - | - |
| | True  | 1025-1280 | 512 | 1024   | 149,755 | 47.49 | 3153.68 |


temperature=0.0
| | model | speculative_model | num_speculative_tokens | Output Tokens | Time (s) | Throughput (tokens/s) |
|---|---|---|---|---|---|---|
| | Qwen3-1.7B |  None       | 0    | 133,966 | 28.38  | 4720.47 |
| | Qwen3-1.7B |  Qwen3-0.6B | 3    | 133,966 | 46.65  | 2871.50 |
| | Qwen3-1.7B |  Qwen3-0.6B | 5    | 133,966 | 44.05  | 3041.23 |
| | Qwen3-4B   |  None       | 0    | 133,966 | 46.99  | 2851.16 |
| | Qwen3-4B   |  Qwen3-0.6B | 3    | 133,966 | 82.84  | 1617.15 |
| | Qwen3-4B   |  Qwen3-0.6B | 5    | 133,966 | 75.36  | 1777.65 |
| | Qwen3-8B   |  None       | 0    | 133,966 | 68.75  | 1948.51 |
| | Qwen3-8B   |  Qwen3-0.6B | 3    | 133,966 | 107.29 | 1248.61 |
| | Qwen3-8B   |  Qwen3-0.6B | 5    | 133,966 | 118.47 | 1130.80 |
