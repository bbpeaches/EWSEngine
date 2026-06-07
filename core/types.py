from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class SliderSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    value: float
    step: float | None = None
    color: str = "slategray"


@dataclass(frozen=True, slots=True)
class RadioSpec:
    key: str
    label: str
    options: tuple[str, ...]
    value: str


@dataclass(frozen=True, slots=True)
class ModuleSpec(Generic[InputT, OutputT]):
    key: str
    name: str
    description: str
    input_model: type[InputT]
    engine: Any
    scene_factory: Callable[[], Any] | None = None
    sliders: tuple[SliderSpec, ...] = field(default_factory=tuple)
    radios: tuple[RadioSpec, ...] = field(default_factory=tuple)
    presets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
