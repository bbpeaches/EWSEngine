from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any

import numpy as np

from backend.interfaces import dataclass_to_mapping
from backend.service import SimulationService
from core.exceptions import FrontendError


class SimulationClient:
    """Single frontend gateway for local and HTTP simulation frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = 3.0
        self._service = SimulationService()

    def fetch_frame(self, module_key: str, payload: dict[str, Any], mode: str) -> dict[str, Any]:
        """Fetch a simulation frame and normalize it to a nested dictionary."""

        normalized_mode = mode.strip().lower()
        if normalized_mode == "local":
            raw_frame = self._fetch_local(module_key, payload)
            return raw_frame
        elif normalized_mode == "http":
            raw_frame = self._fetch_http(module_key, payload)
            return _normalize_mapping(raw_frame)
        else:
            raise FrontendError(f"Unsupported simulation mode: {mode!r}.")

    def _fetch_local(self, module_key: str, payload: Mapping[str, Any]) -> Any:
        try:
            result = self._service.simulate(module_key, payload)
        except Exception as exc:  # noqa: BLE001 - keep the UI boundary resilient.
            raise FrontendError(f"Local simulation failed for '{module_key}': {exc}") from exc
        if is_dataclass(result):
            return dataclass_to_mapping(result)
        return result

    def _fetch_http(self, module_key: str, payload: Mapping[str, Any]) -> Any:
        url = f"http://{self.host}:{self.port}/simulate/{module_key}"
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = _http_error_message(exc)
            raise FrontendError(f"HTTP simulation failed ({exc.code}): {message}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise FrontendError(f"Cannot reach simulation API at {url}: {reason}") from exc
        except TimeoutError as exc:
            raise FrontendError(f"Simulation API timed out at {url}.") from exc

        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise FrontendError(f"Simulation API returned invalid JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise FrontendError("Simulation API response must be a JSON object.")
        if "error" in decoded:
            raise FrontendError(str(decoded["error"]))
        return decoded.get("result", decoded)


def _normalize_mapping(value: Any) -> dict[str, Any]:
    normalized = _normalize_value(value)
    if not isinstance(normalized, dict):
        raise FrontendError(f"Simulation frame must normalize to dict, got {type(value)!r}.")
    return normalized


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_value(dataclass_to_mapping(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw_body = exc.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - preserve the original HTTP error path.
        return str(exc.reason or "unknown HTTP error")
    try:
        decoded = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body or str(exc.reason or "unknown HTTP error")
    if isinstance(decoded, Mapping) and "error" in decoded:
        return str(decoded["error"])
    return raw_body or str(exc.reason or "unknown HTTP error")
