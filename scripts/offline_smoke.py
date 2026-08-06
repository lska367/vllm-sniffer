"""Offline smoke test: LLM.generate with the sniffer plugin loaded."""
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=2048,
    gpu_memory_utilization=0.6,
    enforce_eager=True,  # skip cudagraph capture for the smoke test
)
outs = llm.generate(
    ["Hello, who are you?", "What is 2+2?"],
    sampling_params=SamplingParams(temperature=0, max_tokens=32),
)
for o in outs:
    print(f"[{o.prompt}] -> {o.outputs[0].text!r}")
print("OK")
