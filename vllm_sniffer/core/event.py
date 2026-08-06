"""Event schema shared by every hook.

Events are msgspec.Structs so serialization is fast and allocation-light.
One JSONL line per event. The ``data`` dict must contain only JSON-encodable
values (str/int/float/bool/None/list/dict of those).

Cross-process correlation keys:
- ``req_id``: vLLM request id (same value in API server / engine core / worker)
- ``step``:  engine iteration index (set by engine-core hooks)
- ``pid``:   process that observed the event
- ``ts_ns``: wall-clock ns (comparable across processes/nodes; used for
  ordering in the analysis layer)
"""

from __future__ import annotations

import os
import time
from typing import Any

import msgspec

SCHEMA_VER = 1

# Event group tags: which part of the vLLM process tree emitted the event.
GROUP_API = "api"        # request lifecycle (AsyncLLMEngine / offline LLM)
GROUP_CORE = "core"      # EngineCore.step / Scheduler (engine core process)
GROUP_WORKER = "worker"  # model forward / sampler (GPU worker process)


class Event(msgspec.Struct, gc=False, frozen=True, kw_only=True):
    schema_ver: int = SCHEMA_VER
    ts_ns: int
    pid: int
    group: str
    type: str
    req_id: str | None = None
    step: int | None = None
    sampled: bool = False
    data: dict[str, Any] = msgspec.field(default_factory=dict)


def make_event(
    group: str,
    type_: str,
    *,
    req_id: str | None = None,
    step: int | None = None,
    sampled: bool = False,
    data: dict[str, Any] | None = None,
) -> Event:
    return Event(
        ts_ns=time.time_ns(),
        pid=os.getpid(),
        group=group,
        type=type_,
        req_id=req_id,
        step=step,
        sampled=sampled,
        data=data or {},
    )


def encode_line(event: Event) -> bytes:
    """Serialize one event to a JSONL line (without trailing newline)."""
    return msgspec.json.encode(event)
