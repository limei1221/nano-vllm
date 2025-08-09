import argparse
import json
import time
import uuid
import threading
from typing import Any, Dict, Generator, List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from nanovllm import LLM, SamplingParams


def create_app(llm: LLM) -> FastAPI:
    app = FastAPI(title="nanoVLLM OpenAI-Compatible Server")
    generation_lock = threading.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        model_id = getattr(llm.tokenizer, "name_or_path", "unknown-model")
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "owner",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: Dict[str, Any] = Body(...)):
        messages: List[Dict[str, str]] = payload.get("messages", [])
        model: Optional[str] = payload.get("model")
        temperature: float = float(payload.get("temperature", 1.0))
        max_tokens: int = int(payload.get("max_tokens", 64))
        stream: bool = bool(payload.get("stream", False))
        n: int = int(payload.get("n", 1))

        if not messages:
            raise HTTPException(status_code=400, detail="'messages' is required")

        try:
            prompt: str = llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=bool(payload.get("ignore_eos", False)),
        )

        created_ts = int(time.time())
        model_id = model or getattr(llm.tokenizer, "name_or_path", "unknown-model")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        if stream:
            # Simulated streaming: generate fully, then emit token deltas
            with generation_lock:
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            output = outputs[0]
            completion_token_ids: List[int] = output["token_ids"]

            def event_gen() -> Generator[bytes, None, None]:
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n".encode("utf-8")

                accumulated: List[int] = []
                for token_id in completion_token_ids:
                    accumulated.append(token_id)
                    piece = llm.tokenizer.decode([token_id], skip_special_tokens=True)
                    if piece:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        # Non-streaming response
        prompts = [prompt] * n
        with generation_lock:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        prompt_token_ids = llm.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)

        choices = []
        for i in range(n):
            item = outputs[i]
            text = item["text"]
            completion_tokens = len(item["token_ids"]) if "token_ids" in item else 0
            choice = {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
            choices.append(choice)

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_id,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": sum(len(o.get("token_ids", [])) for o in outputs),
                "total_tokens": prompt_tokens + sum(len(o.get("token_ids", [])) for o in outputs),
            },
        }
        return JSONResponse(content=response)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoVLLM OpenAI-compatible server")
    parser.add_argument("--model", type=str, required=True, help="Path to the HF model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine_kwargs: Dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.speculative_model:
        engine_kwargs["speculative_model"] = args.speculative_model
        engine_kwargs["num_speculative_tokens"] = args.num_speculative_tokens

    llm = LLM(args.model, **engine_kwargs)

    app = create_app(llm)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
