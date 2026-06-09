from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.service import ensure_registry
from frontend.app_v2 import APP_VERSION, ModernAppBase
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


def test_axis_view_presets_change_camera() -> None:
    scene = TemScene()
    try:
        scene.apply_view_preset("+Z 投影")
        assert scene.ax.elev == 90.0
        scene.apply_view_preset("-X 投影")
        assert scene.ax.azim == 180.0
        scene.apply_view_preset("默认")
        assert scene.ax.elev == scene.default_elev
        assert scene.ax.azim == scene.default_azim
    finally:
        plt.close(scene.fig)


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakePresetApp(ModernAppBase):
    def __init__(self) -> None:
        self.scene = WaveScene()
        self.slider_vars = {spec.key: FakeVar(spec.value) for spec in self.scene.slider_specs}
        self.radio_vars = {spec.key: FakeVar(spec.value) for spec in self.scene.radio_specs}
        self.radio_specs = {spec.key: spec for spec in self.scene.radio_specs}
        self.preset_var = FakeVar("空气行波")


def test_preset_selection_clears_when_controls_diverge() -> None:
    app = FakePresetApp()
    try:
        app.radio_vars["mode"].set("lossy")
        app._sync_preset_selection_from_controls()
        assert app.preset_var.get() == ""
        app.radio_vars["mode"].set("material")
        app.radio_vars["material"].set("真空/空气 (εr=1.0)")
        app.slider_vars["freq_mhz"].set(1000.0)
        app._sync_preset_selection_from_controls()
        assert app.preset_var.get() == "空气行波"
    finally:
        plt.close(app.scene.fig)


def test_tem_scene_defaults_to_377h() -> None:
    h_spec = next(spec for spec in TemScene.radio_specs if spec.key == "h_display")
    assert h_spec.value == "377H"


def test_app_version_and_zoom_label() -> None:
    app = object.__new__(ModernAppBase)
    app.zoom = 1.0
    assert APP_VERSION == "0.2.3"
    assert app._format_zoom() == "缩放 1.00x"


def ensure_modules():
    from core.registry import registry

    return registry.all()
