"""API hook tests: request lifecycle with fake vLLM classes."""

import asyncio
from types import SimpleNamespace

from conftest import fake_vllm_module

from vllm_sniffer.hooks.api import install_api_hooks


def _request_output(token_ids, finished=False, finish_reason=None, prompt_tokens=5):
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
        finished=finished,
        finish_reason=finish_reason,
        prompt_token_ids=[0] * prompt_tokens,
    )


def test_async_generate_lifecycle(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.async_llm")

    class FakeAsyncLLMEngine:
        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            yield _request_output([])  # prompt echo, no token yet
            yield _request_output([1], finished=True, finish_reason=SimpleNamespace(value="stop"))

    class FakeAsyncLLM(FakeAsyncLLMEngine):
        pass

    mod.AsyncLLM = FakeAsyncLLM
    mod.AsyncLLMEngine = FakeAsyncLLMEngine
    install_api_hooks()

    async def run():
        engine = FakeAsyncLLM()
        gen = engine.generate("hello", None, "req-1")
        async for _ in gen:
            pass

    asyncio.run(run())

    events = read_events()
    types = [e["type"] for e in events]
    assert types == ["request_start", "request_first_token", "request_finish"]
    assert all(e["req_id"] == "req-1" for e in events)
    start = events[0]
    assert start["data"]["kind"] == "str"  # prompt never stored, only shape
    assert events[1]["data"]["n_prompt_tokens"] == 5
    assert events[2]["data"]["finish_reason"] == "stop"
    assert events[2]["data"]["n_output_tokens"] == 1


def test_async_abort_hook(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.async_llm")

    class FakeAsyncLLM:
        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            yield _request_output([1])

        async def abort(self, request_id, internal=False):
            pass

    mod.AsyncLLM = FakeAsyncLLM
    install_api_hooks()

    async def run():
        await FakeAsyncLLM().abort("req-9")

    asyncio.run(run())
    events = read_events()
    aborts = [e for e in events if e["type"] == "request_abort"]
    assert len(aborts) == 1
    assert aborts[0]["req_id"] == "req-9"
    assert aborts[0]["data"]["source"] == "explicit_abort"


def test_async_generate_abort_on_client_close(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.async_llm")

    class FakeAsyncLLMEngine:
        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            yield _request_output([1])
            yield _request_output([2])  # never consumed -> client closed

    mod.AsyncLLMEngine = FakeAsyncLLMEngine
    install_api_hooks()

    async def run():
        engine = FakeAsyncLLMEngine()
        gen = engine.generate("hello", None, "req-2")
        await gen.__anext__()  # consume one token, then close the stream
        await gen.aclose()

    asyncio.run(run())

    events = read_events()
    types = [e["type"] for e in events]
    assert types == ["request_start", "request_first_token", "request_abort"]
    assert events[2]["req_id"] == "req-2"


def test_async_generate_engine_error_propagates(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.async_llm")

    class FakeAsyncLLMEngine:
        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            raise RuntimeError("engine exploded")
            yield  # pragma: no cover - makes this an async generator

    mod.AsyncLLMEngine = FakeAsyncLLMEngine
    install_api_hooks()

    async def run():
        engine = FakeAsyncLLMEngine()
        async for _ in engine.generate("hello", None, "req-3"):
            pass

    try:
        asyncio.run(run())
        raised = False
    except RuntimeError:
        raised = True
    assert raised  # the error must reach the caller unchanged

    events = read_events()
    assert [e["type"] for e in events] == ["request_start", "request_abort"]


def test_offline_generate(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.entrypoints.llm")

    class FakeLLM:
        def generate(self, prompts, sampling_params=None):
            return ["out1", "out2"]

    mod.LLM = FakeLLM
    install_api_hooks()

    FakeLLM().generate(["p1", "p2"])

    events = read_events()
    assert [e["type"] for e in events] == ["request_start", "request_finish"]
    assert events[0]["data"]["mode"] == "offline"
    assert events[0]["data"]["n_requests"] == 2
    assert events[1]["data"]["n_finished"] == 2
    assert events[1]["data"]["dur_ns"] >= 0
