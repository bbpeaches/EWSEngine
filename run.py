from __future__ import annotations

import argparse
import socket
import threading
import time

from api.server import run_server
from frontend.launcher import run_desktop_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EWSEngine desktop app and local API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--desktop-only", action="store_true")
    return parser.parse_args()


def wait_for_server(host: str, port: int, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.05):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"API server did not become ready on {host}:{port}{detail}")


def main() -> None:
    args = parse_args()
    if args.desktop_only:
        run_desktop_app(host=args.host, port=args.port)
        return
    if args.api_only:
        run_server(host=args.host, port=args.port)
        return

    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": args.host, "port": args.port},
        daemon=True,
    )
    server_thread.start()
    wait_for_server(args.host, args.port)
    run_desktop_app(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
