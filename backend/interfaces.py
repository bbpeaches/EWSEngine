from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

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
        return {field.name: getattr(model, field.name) for field in fields(model)}
    raise TypeError(f"Expected dataclass instance, got {type(model)!r}.")


def to_mapping_tree(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_mapping_tree(item) for key, item in dataclass_to_mapping(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_mapping_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_mapping_tree(item) for item in value)
    if isinstance(value, list):
        return [to_mapping_tree(item) for item in value]
    return value


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in dataclass_to_mapping(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
