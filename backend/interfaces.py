from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class SerializableModel(Protocol):
    def __dict__(self) -> dict[str, Any]:
        ...


class SimulationEngine(ABC, Generic[InputT, OutputT]):
    """Typed simulation contract shared by the API and desktop frontend."""

    @abstractmethod
    def simulate(self, input_data: InputT) -> OutputT:
        raise NotImplementedError


def dataclass_to_mapping(model: Any) -> Mapping[str, Any]:
    if is_dataclass(model):
        return asdict(model)
    raise TypeError(f"Expected dataclass instance, got {type(model)!r}.")
