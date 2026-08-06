"""Event schema + JSONL encoding tests."""

from vllm_sniffer.core.event import Event, encode_line, make_event


def test_make_event_defaults():
    ev = make_event("api", "request_start", req_id="r1")
    assert ev.group == "api"
    assert ev.type == "request_start"
    assert ev.req_id == "r1"
    assert ev.step is None
    assert ev.sampled is False
    assert ev.data == {}
    assert ev.ts_ns > 0
    assert ev.pid > 0


def test_encode_line_is_json():
    ev = make_event(
        "worker",
        "sample_flip",
        data={"margin": 0.0005, "top1": 42, "top2": 7},
    )
    line = encode_line(ev)
    assert line.endswith(b"}") or line.endswith(b"}\n")
    assert b'"sample_flip"' in line
    assert b'"margin":0.0005' in line.replace(b" ", b"")


def test_decode_roundtrip():
    import msgspec

    ev = make_event("core", "preempt", req_id="r2", data={"mode": "recompute"})
    decoded = msgspec.json.decode(encode_line(ev), type=Event)
    assert decoded == ev
