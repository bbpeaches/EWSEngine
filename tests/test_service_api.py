from __future__ import annotations

import json
import threading
import urllib.request

from api.server import create_server
from backend.service import SimulationService


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
