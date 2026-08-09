#!/usr/bin/env python3
"""Run the vllm-sniffer visualization server (roadmap P2-1).

Usage:
    python -m webapp.server --dir /tmp/vllm-sniffer --port 8080

Requires the optional dependencies: uv pip install -e '.[web]'
Open http://127.0.0.1:<port> -- pick a run_id and see four views:
request timeline / step duration / flip heatmap / batch-vs-latency scatter.
API documentation: doc/WEBAPP.md
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp import create_app  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None,
                    help="event root dir (default: $VLLM_SNIFFER_DIR or "
                         "/tmp/vllm-sniffer)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError as e:
        print(
            "webapp requires the optional 'web' dependencies:\n"
            "  uv pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 2

    app = create_app(args.dir or os.environ.get("VLLM_SNIFFER_DIR"))
    print("vllm-sniffer viz: http://{}:{}".format(args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
