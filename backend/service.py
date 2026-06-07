from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from backend.models import (
    OpticsInput,
    PolarizationInput,
    SpeedInput,
    TemInput,
    TransmissionInput,
    WaveInput,
)
from backend.physics.optics import OpticsEngine
from backend.physics.polarization import PolarizationEngine
from backend.physics.speed import SpeedEngine
from backend.physics.tem import TemEngine
from backend.physics.transmission import TransmissionEngine
from backend.physics.wave import MATERIALS, WaveEngine
from core.registry import registry
from core.types import JsonDict, ModuleSpec, RadioSpec, SliderSpec


class SimulationService:
    """Shared simulation service used by the local API and desktop frontend."""

    def __init__(self) -> None:
        ensure_registry(include_scenes=False)

    def list_modules(self) -> tuple[ModuleSpec[Any, Any], ...]:
        return registry.all()

    def get_module(self, key: str) -> ModuleSpec[Any, Any]:
        return registry.get(key)

    def simulate(self, key: str, payload: Mapping[str, Any] | None = None) -> Any:
        spec = self.get_module(key)
        model = spec.input_model(**dict(payload or {}))
        return spec.engine.simulate(model)


def ensure_registry(*, include_scenes: bool) -> None:
    if registry.all() and (not include_scenes or all(spec.scene_factory is not None for spec in registry.all())):
        return

    specs = build_specs(include_scenes=include_scenes)
    registry.clear()
    for spec in specs:
        registry.register(spec, replace=True)


def build_specs(*, include_scenes: bool) -> tuple[ModuleSpec[Any, Any], ...]:
    return (
        ModuleSpec(
            key="optics",
            name="界面光学与共面性",
            description="Reflection, refraction, total internal reflection, Brewster angle, and coplanarity.",
            input_model=OpticsInput,
            engine=OpticsEngine(),
            scene_factory=scene_factory("optics") if include_scenes else None,
            sliders=(
                SliderSpec("n1", "入射媒质 n1", 1.0, 4.0, 1.0, 0.01),
                SliderSpec("n2", "透射媒质 n2", 1.0, 4.0, 1.5, 0.01),
                SliderSpec("theta_deg", "入射角 theta_i", 0.0, 89.9, 40.0, 0.1, "royalblue"),
                SliderSpec("phi_deg", "入射面方位角 phi", 0.0, 360.0, 0.0, 1.0, "darkorange"),
            ),
            radios=(RadioSpec("polarization", "偏振模式", ("natural", "s", "p"), "natural"),),
            presets={
                "空气 -> 玻璃": {"n1": 1.0, "n2": 1.5, "theta_deg": 40.0, "phi_deg": 0.0},
                "玻璃 -> 空气": {"n1": 1.5, "n2": 1.0, "theta_deg": 45.0, "phi_deg": 0.0},
                "水 -> 空气": {"n1": 1.33, "n2": 1.0, "theta_deg": 50.0, "phi_deg": 35.0},
            },
        ),
        ModuleSpec(
            key="polarization",
            name="极化合成",
            description="Phase, circular-basis, and antenna-match polarization synthesis.",
            input_model=PolarizationInput,
            engine=PolarizationEngine(),
            scene_factory=scene_factory("polarization") if include_scenes else None,
            sliders=(
                SliderSpec("p1", "参数 1", 0.0, 1.5, 1.0, 0.05, "royalblue"),
                SliderSpec("p2", "参数 2", 0.0, 1.5, 1.0, 0.05, "forestgreen"),
                SliderSpec("p3", "参数 3", -180.0, 180.0, 90.0, 1.0, "darkorange"),
            ),
            radios=(RadioSpec("mode", "实验模式", ("phase", "circular", "match"), "phase"),),
        ),
        ModuleSpec(
            key="transmission",
            name="行驻波传播",
            description="Standing-wave envelope and VSWR decomposition.",
            input_model=TransmissionInput,
            engine=TransmissionEngine(),
            scene_factory=scene_factory("transmission") if include_scenes else None,
            sliders=(SliderSpec("reflection_coefficient", "反射系数 R", -0.99, 0.99, 0.0, 0.01),),
            radios=(RadioSpec("mode", "实验模式", ("vswr", "standing"), "vswr"),),
        ),
        ModuleSpec(
            key="wave",
            name="基础波动",
            description="Material wave, lossy medium, and constant-phase planes.",
            input_model=WaveInput,
            engine=WaveEngine(),
            scene_factory=scene_factory("wave") if include_scenes else None,
            sliders=(
                SliderSpec("freq_mhz", "频率 MHz", 100.0, 3000.0, 1000.0, 10.0, "royalblue"),
                SliderSpec("alpha", "衰减常数 alpha", 0.0, 1.0, 0.3, 0.01),
                SliderSpec("beta", "相位常数 beta", 1.0, 10.0, 5.0, 0.05, "forestgreen"),
                SliderSpec("theta_deg", "极角 theta", 0.0, 180.0, 45.0, 1.0, "royalblue"),
                SliderSpec("phi_deg", "方位角 phi", 0.0, 360.0, 45.0, 1.0, "darkorange"),
                SliderSpec("spacing", "面间距 lambda", 0.5, 4.0, 1.5, 0.05, "mediumpurple"),
            ),
            radios=(
                RadioSpec("mode", "实验模式", ("material", "lossy", "planes"), "material"),
                RadioSpec("material", "材料", tuple(MATERIALS.keys()), next(iter(MATERIALS.keys()))),
            ),
        ),
        ModuleSpec(
            key="tem",
            name="TEM 平面波",
            description="Uniform TEM wave with propagation-axis and polarity controls.",
            input_model=TemInput,
            engine=TemEngine(),
            scene_factory=scene_factory("tem") if include_scenes else None,
            sliders=(
                SliderSpec("amplitude", "振幅", 0.5, 6.0, 3.0, 0.1),
                SliderSpec("wavelength", "波长", 1.0, 10.0, 5.0, 0.1),
                SliderSpec("speed", "传播速度", 0.5, 5.0, 2.0, 0.1),
                SliderSpec("time_scale", "动画速度", 0.2, 3.0, 1.0, 0.1),
            ),
            radios=(
                RadioSpec("direction", "传播轴", ("x", "y", "z"), "x"),
                RadioSpec("polarity", "相位方向", ("1", "-1"), "1"),
            ),
        ),
        ModuleSpec(
            key="speed",
            name="波速效应",
            description="Dispersion and apparent phase-speed geometry.",
            input_model=SpeedInput,
            engine=SpeedEngine(),
            scene_factory=scene_factory("speed") if include_scenes else None,
            sliders=(
                SliderSpec("vg", "群速 Vg", 0.1, 2.5, 0.5, 0.01, "firebrick"),
                SliderSpec("theta_deg", "观察夹角 theta", 0.0, 85.0, 45.0, 1.0, "forestgreen"),
                SliderSpec("amplitude", "振幅", 0.4, 2.4, 1.0, 0.05, "royalblue"),
                SliderSpec("carrier_lambda", "载波波长 lambda", 1.2, 5.0, 2.0, 0.05, "darkorange"),
            ),
            radios=(
                RadioSpec("mode", "实验模式", ("dispersion", "apparent"), "dispersion"),
                RadioSpec("direction", "传播方向", ("x", "y", "z"), "z"),
            ),
        ),
    )


def scene_factory(key: str):
    def resolve() -> type[Any]:
        if key == "optics":
            from frontend.scenes.optics import OpticsScene

            return OpticsScene
        if key == "polarization":
            from frontend.scenes.polarization import PolarizationScene

            return PolarizationScene
        if key == "transmission":
            from frontend.scenes.transmission import TransmissionScene

            return TransmissionScene
        if key == "wave":
            from frontend.scenes.wave import WaveScene

            return WaveScene
        if key == "tem":
            from frontend.scenes.tem import TemScene

            return TemScene
        if key == "speed":
            from frontend.scenes.speed import SpeedScene

            return SpeedScene
        raise KeyError(key)

    return resolve


def module_summary(spec: ModuleSpec[Any, Any]) -> JsonDict:
    return {
        "key": spec.key,
        "name": spec.name,
        "description": spec.description,
        "presets": {name: dict(values) for name, values in spec.presets.items()},
        "sliders": [asdict(slider) for slider in spec.sliders],
        "radios": [asdict(radio) for radio in spec.radios],
    }
