import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def non_streaming_request(base_url: str, messages: List[Dict[str, str]], model: str, **kwargs: Any) -> None:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {"messages": messages, "model": model}
    payload.update(kwargs)
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))


def streaming_request(base_url: str, messages: List[Dict[str, str]], model: str, **kwargs: Any) -> None:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {"messages": messages, "model": model, "stream": True}
    payload.update(kwargs)

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                print()
                break
            try:
                event = json.loads(data)
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for nanoVLLM OpenAI-compatible server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="local-model")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--message", type=str, default="Introduce yourself")

    args = parser.parse_args()

    messages = [
        {"role": "user", "content": args.message},
    ]

    if args.stream:
        streaming_request(
            args.base_url,
            messages,
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        non_streaming_request(
            args.base_url,
            messages,
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
