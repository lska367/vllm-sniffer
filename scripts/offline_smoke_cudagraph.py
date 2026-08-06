"""Smoke test with default settings: torch.compile + cudagraph enabled."""
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=2048,
    gpu_memory_utilization=0.6,
)
outs = llm.generate(
    ["Hello, who are you?"],
    sampling_params=SamplingParams(temperature=0, max_tokens=64),
)
print(f"-> {outs[0].outputs[0].text!r}")
print("OK")
