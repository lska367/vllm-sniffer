"""Visualization frontend tests (P2-1): FastAPI aggregation endpoints.

The API layer is exercised with starlette's TestClient against a synthetic
run directory (same builder as test_tools.py). Frontend assets are checked
for the four views' canvas ids. Requires the optional 'web' deps
(fastapi + httpx); skipped when absent.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

fastapi = pytest.importorskip("fastapi")

from test_tools import make_synthetic_run  # noqa: E402

from webapp import create_app  # noqa: E402

RUN_ID = "run-20260809-120000"


@pytest.fixture
def client(tmp_path):
    make_synthetic_run(tmp_path)
    app = create_app(out_dir=str(tmp_path))
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_runs_list(client):
    data = client.get("/api/runs").json()
    runs = data["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == RUN_ID
    assert runs[0]["kind"] == "dir"
    assert runs[0]["n_events"] == 12
    assert runs[0]["types"]["step"] == 2


def test_summary(client):
    data = client.get(f"/api/runs/{RUN_ID}/summary").json()
    assert data["n_events"] == 12
    assert data["types"]["request_start"] == 2
    assert data["types"]["sample_flip"] == 1
    assert data["env"]["python_version"] == "3.12"  # from env_snapshot


def test_timeline(client):
    data = client.get(f"/api/runs/{RUN_ID}/timeline").json()
    reqs = data["requests"]
    assert [r["req_id"] for r in reqs] == ["req-001", "req-002"]
    r1 = reqs[0]
    assert abs(r1["ttft_ms"] - 50.0) < 1e-6
    assert abs(r1["tpot_ms"] - (650.0 / 63.0)) < 1e-3
    assert r1["n_output_tokens"] == 64
    assert r1["finish_reason"] == "length"
    assert r1["aborted"] is False


def test_steps(client):
    data = client.get(f"/api/runs/{RUN_ID}/steps").json()
    steps = data["steps"]
    assert [s["step"] for s in steps] == [1, 2]
    assert steps[0]["dur_ms"] == 8.0
    assert steps[1]["dur_ms"] == 12.0
    assert steps[0]["n_outputs"] == 1


def test_flips(client):
    data = client.get(f"/api/runs/{RUN_ID}/flips").json()
    assert data["total"] == 1
    f = data["flips"][0]
    assert f["margin"] == 0.0004
    assert f["top1"] == 12 and f["top2"] == 7


def test_scatter(client):
    data = client.get(f"/api/runs/{RUN_ID}/scatter").json()
    assert len(data["points"]) == 1
    p = data["points"][0]
    assert p["num_tokens"] == 42
    assert p["num_seqs"] == 1
    assert p["dur_ms"] == 7.0


def test_unknown_run_404(client):
    r = client.get("/api/runs/nope/summary")
    assert r.status_code == 404
    assert "run not found" in r.json()["detail"]


def test_path_traversal_rejected(client):
    for bad in ("..", "..%2F..", "%2e%2e", "a/b"):
        r = client.get(f"/api/runs/{bad}/summary")
        assert r.status_code == 404, bad


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "vllm-sniffer" in html
    for cv in ("timeline", "steps", "flips", "scatter"):
        assert f'id="{cv}"' in html


def test_healthz(client):
    assert client.get("/healthz").json()["ok"] is True


def test_parquet_run_listed(tmp_path):
    pytest.importorskip("pyarrow")
    run = make_synthetic_run(tmp_path)
    out = tmp_path / "exported.parquet"
    r = subprocess.run(
        [sys.executable, "tools/export_parquet.py", str(run), "-o", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr

    from fastapi.testclient import TestClient

    app = create_app(out_dir=str(tmp_path))
    with TestClient(app) as c:
        runs = c.get("/api/runs").json()["runs"]
        assert len(runs) == 2
        pq_run = next(r for r in runs if r["kind"] == "parquet")
        assert pq_run["run_id"] == "exported"
        assert pq_run["n_events"] == 12
        assert c.get("/api/runs/exported/timeline").status_code == 200
