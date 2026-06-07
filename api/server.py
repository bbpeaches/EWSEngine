from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np

from backend.service import SimulationService, ensure_registry, module_summary
from core.exceptions import ApiError, EWSEError, RegistryError, ValidationError


class SimulationRequestHandler(BaseHTTPRequestHandler):
    service = SimulationService()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/modules":
            ensure_registry(include_scenes=False)
            modules = [module_summary(spec) for spec in self.service.list_modules()]
            self._send_json({"modules": modules})
            return
        if parsed.path.startswith("/modules/"):
            key = parsed.path.rsplit("/", 1)[-1]
            spec = self.service.get_module(key)
            self._send_json(module_summary(spec))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/simulate/"):
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        key = parsed.path.rsplit("/", 1)[-1]
        payload = self._read_json()
        result = self.service.simulate(key, payload)
        self._send_json({"result": to_jsonable(result)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError("JSON body must be an object.")
        return data

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except RegistryError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValidationError, ApiError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except EWSEError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    ensure_registry(include_scenes=False)
    return ThreadingHTTPServer((host, port), SimulationRequestHandler)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_server(host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
