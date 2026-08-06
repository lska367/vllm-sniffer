"""Online smoke test: streaming completions + client-abort scenario.

Verifies the streaming path events:
- request_start / request_first_token / request_finish for a normal request
- request_abort when the client disconnects mid-stream
"""

import json
import sys

import httpx

URL = "http://localhost:8099/v1/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def stream_full(prompt: str, max_tokens: int = 64) -> list[str]:
    """Read a streaming response to completion; return SSE data payloads."""
    payloads: list[str] = []
    with httpx.stream(
        "POST",
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        timeout=None,
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                payloads.append(line[6:])
    return payloads


def stream_then_abort(prompt: str, n_chunks: int = 3) -> None:
    """Read a few chunks, then drop the connection (simulated client abort)."""
    with httpx.stream(
        "POST",
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0,
            "stream": True,
        },
        timeout=None,
    ) as r:
        for i, line in enumerate(r.iter_lines()):
            if line.startswith("data: "):
                print(f"  chunk {i}: {line[6:][:60]}...")
                if i >= n_chunks:
                    print(f"  -> client disconnects after {i} chunks")
                    return  # closing the context = abort


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("full", "both"):
        print("=== normal streaming request ===")
        payloads = stream_full("Explain the concept of gravity in one sentence.")
        chunks = []
        for p in payloads:
            if p == "[DONE]":
                continue
            data = json.loads(p)
            if data.get("choices") and data["choices"][0].get("text"):
                chunks.append(data["choices"][0]["text"])
        text = "".join(chunks)
        print(f"  streamed {len(payloads)} chunks, output={text!r}")
        assert payloads and any(
            '"finish_reason"' in p and '"finish_reason":null' not in p for p in payloads
        )
        print("  OK: stream completed with a finish_reason")

    if mode in ("abort", "both"):
        print("=== abort scenario (client disconnects mid-stream) ===")
        stream_then_abort("Write a very long story about a dragon.", n_chunks=3)
        print("  OK: client disconnected")
