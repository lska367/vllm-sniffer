"""Engine-core hook tests: EngineCore.step and Scheduler.schedule."""

from types import SimpleNamespace

from conftest import fake_vllm_module

from vllm_sniffer.hooks.engine import install_engine_hooks


def _fake_outputs(n):
    return {0: SimpleNamespace(outputs=[object() for _ in range(n)])}


def test_step_event(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.core")

    class FakeEngineCore:
        def step(self):
            return _fake_outputs(3), False

    mod.EngineCore = FakeEngineCore
    install_engine_hooks()

    FakeEngineCore().step()

    events = read_events()
    step_events = [e for e in events if e["type"] == "step"]
    assert len(step_events) == 1
    assert step_events[0]["group"] == "core"
    assert step_events[0]["data"]["n_outputs"] == 3
    assert step_events[0]["data"]["dur_ns"] >= 0


def test_schedule_with_preemption(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.core.sched.scheduler")

    class FakeScheduler:
        def schedule(self):
            return SimpleNamespace(
                preempted_req_ids={"r1", "r2"},
                total_num_scheduled_tokens=100,
                num_scheduled_tokens={"r3": 60, "r4": 40},
                finished_req_ids={"r9"},
                new_block_ids_to_zero=[1, 2, 3],
            )

    mod.Scheduler = FakeScheduler
    install_engine_hooks()

    FakeScheduler().schedule()

    events = read_events()
    preempts = [e for e in events if e["type"] == "preempt"]
    assert sorted(e["req_id"] for e in preempts) == ["r1", "r2"]
    assert all(e["data"]["mode"] == "recompute" for e in preempts)

    sched = [e for e in events if e["type"] == "schedule"][0]
    assert sched["sampled"] is True
    assert sched["data"]["total_num_scheduled_tokens"] == 100
    assert sched["data"]["n_scheduled_reqs"] == 2
    assert sched["data"]["n_preempted"] == 2
    assert sched["data"]["finished_req_ids"] == 1
    assert sched["data"]["new_block_ids_to_zero"] == 3


def test_schedule_without_preemption_emits_no_preempt(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.core.sched.scheduler")

    class FakeScheduler:
        def schedule(self):
            return SimpleNamespace(
                preempted_req_ids=None,
                total_num_scheduled_tokens=10,
                num_scheduled_tokens={"r3": 10},
                finished_req_ids=set(),
                new_block_ids_to_zero=None,
            )

    mod.Scheduler = FakeScheduler
    install_engine_hooks()

    FakeScheduler().schedule()

    events = read_events()
    assert all(e["type"] != "preempt" for e in events)
    sched = [e for e in events if e["type"] == "schedule"][0]
    assert sched["data"]["n_preempted"] == 0


def test_step_exception_propagates(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.engine.core")

    class FakeEngineCore:
        def step(self):
            raise RuntimeError("boom")

    mod.EngineCore = FakeEngineCore
    install_engine_hooks()

    try:
        FakeEngineCore().step()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert read_events() == []  # nothing recorded for the failed step
