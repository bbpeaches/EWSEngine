from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.service import ensure_registry
from frontend.scenes.optics import OpticsScene
from frontend.scenes.polarization import PolarizationScene
from frontend.scenes.speed import SpeedScene
from frontend.scenes.tem import TemScene
from frontend.scenes.transmission import TransmissionScene
from frontend.scenes.wave import WaveScene


def test_all_frontend_scenes_instantiate_under_agg() -> None:
    scenes = [OpticsScene, PolarizationScene, TransmissionScene, WaveScene, TemScene, SpeedScene]
    created = []
    try:
        for scene_type in scenes:
            scene = scene_type()
            created.append(scene)
            assert scene.fig is not None
            assert scene.ax is not None
            assert scene.title
    finally:
        for scene in created:
            plt.close(scene.fig)


def test_registry_can_bind_frontend_scene_factories() -> None:
    ensure_registry(include_scenes=True)
    modules = {module.key: module for module in ensure_modules()}
    assert modules["optics"].scene_factory is not None
    assert modules["polarization"].scene_factory is not None
    assert modules["speed"].scene_factory is not None


def ensure_modules():
    from core.registry import registry

    return registry.all()
