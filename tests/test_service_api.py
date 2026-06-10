from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pytest

from api.server import create_server
from backend.interfaces import dataclass_to_mapping, to_jsonable
from backend.models import LineData
from backend.service import SimulationService
from core.exceptions import ValidationError


def test_service_lists_all_registered_modules() -> None:
    service = SimulationService()
    keys = {module.key for module in service.list_modules()}
    assert keys == {"optics", "polarization", "transmission", "wave", "tem", "speed"}


def test_service_simulates_each_module_with_defaults() -> None:
    service = SimulationService()
    for key in ("optics", "polarization", "transmission", "wave", "tem", "speed"):
        result = service.simulate(key)
        assert result is not None
    assert service.simulate("tem").h_display == "377H"


def test_service_supports_lossy_tem_and_wave_magnetic_line() -> None:
    service = SimulationService()
    tem_result = service.simulate(
        "tem",
        {"mode": "lossy", "direction": "z", "polarity": -1.0, "alpha": 0.3, "beta": 6.0, "speed": 2.0},
    )
    wave_result = service.simulate("wave", {"mode": "lossy", "alpha": 0.3, "beta": 5.0})
    assert len(tem_result.electric_line.x) > 0
    assert len(tem_result.magnetic_line.x) > 0
    assert len(wave_result.magnetic_line.x) > 0


def test_h_display_radio_is_exposed_for_magnetic_scenes() -> None:
    service = SimulationService()
    for key in ("polarization", "transmission", "wave", "tem"):
        radios = {radio.key: radio for radio in service.get_module(key).radios}
        assert radios["h_display"].options == ("隐藏", "H", "377H")
        expected = "377H" if key == "tem" else "隐藏"
        assert radios["h_display"].value == expected


def test_service_rejects_invalid_payloads() -> None:
    service = SimulationService()
    with pytest.raises(ValidationError, match="Unknown input field"):
        service.simulate("optics", {"theta_deg": 35.0, "surprise": 1.0})
    with pytest.raises(ValidationError, match="numeric"):
        service.simulate("optics", {"theta_deg": "wide"})
    with pytest.raises(ValidationError, match="Invalid value"):
        service.simulate("wave", {"mode": "unknown"})


def test_service_coerces_radio_and_numeric_payloads() -> None:
    service = SimulationService()
    frame = service.simulate("tem", {"polarity": "-1", "speed": "2.5"})
    assert frame is not None


@dataclass(frozen=True, slots=True)
class ArrayEnvelope:
    line: LineData
    scalar: np.float64


def test_shared_serializer_preserves_local_arrays_and_normalizes_json() -> None:
    line = LineData(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]), np.asarray([5.0, 6.0]))
    envelope = ArrayEnvelope(line=line, scalar=np.float64(7.5))

    shallow = dataclass_to_mapping(envelope)
    assert shallow["line"] is line
    assert to_jsonable(envelope) == {
        "line": {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]},
        "scalar": 7.5,
    }


def test_api_health_modules_and_simulation_endpoints() -> None:
    server = create_server(port=0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health == {"status": "ok"}

        with urllib.request.urlopen(f"http://{host}:{port}/modules", timeout=3) as response:
            modules = json.loads(response.read().decode("utf-8"))["modules"]
        assert {module["key"] for module in modules} >= {"optics", "tem", "speed"}

        request = urllib.request.Request(
            f"http://{host}:{port}/simulate/optics",
            data=json.dumps({"n1": 1.0, "n2": 1.5, "theta_deg": 35.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))["result"]
        assert result["result"]["is_tir"] is False
        assert result["result"]["R_s"] > 0.0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _start_test_server():
    server = create_server(port=0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, host, port


def _post_json(url: str, data: bytes, headers: dict[str, str] | None = None) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_api_rejects_bad_simulation_requests() -> None:
    server, thread, host, port = _start_test_server()
    try:
        url = f"http://{host}:{port}/simulate/optics"
        cases = (
            (b"{", "Invalid JSON body"),
            (json.dumps(["not", "object"]).encode("utf-8"), "JSON body must be an object"),
            (json.dumps({"theta_deg": "wide"}).encode("utf-8"), "must be numeric"),
            (json.dumps({"polarization": "q"}).encode("utf-8"), "Invalid value"),
            (json.dumps({"theta_deg": 35.0, "extra": 1.0}).encode("utf-8"), "Unknown input field"),
        )
        for data, expected in cases:
            status, payload = _post_json(url, data)
            assert status == 400
            assert expected in str(payload["error"])

        status, payload = _post_json(f"http://{host}:{port}/simulate/missing", b"{}")
        assert status == 404
        assert "not registered" in str(payload["error"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_api_rejects_invalid_content_length() -> None:
    server, thread, host, port = _start_test_server()
    try:
        with socket.create_connection((host, port), timeout=3) as sock:
            request = (
                "POST /simulate/optics HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: invalid\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{}"
            )
            sock.sendall(request.encode("utf-8"))
            response = sock.recv(4096).decode("utf-8", errors="replace")
        assert "400 Bad Request" in response
        assert "Content-Length" in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
