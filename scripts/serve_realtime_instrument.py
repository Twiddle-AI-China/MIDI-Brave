from __future__ import annotations

import argparse
from pathlib import Path

from aiohttp import web

from midibrave.realtime_engine import RealtimeEngine
from midibrave.realtime_server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the local MidiBrave realtime instrument",
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RealtimeEngine.load(args.runtime)
    app = create_app(engine, args.root)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
