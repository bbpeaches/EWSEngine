from __future__ import annotations

from typing import Any, Generic, TypeVar

from core.exceptions import RegistryError
from core.types import ModuleSpec

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class SingletonMeta(type):
    """Metaclass used for process-wide singletons."""

    _instances: dict[type[Any], Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SimulationRegistry(Generic[InputT, OutputT], metaclass=SingletonMeta):
    """Typed registry for simulation modules and scenes."""

    def __init__(self) -> None:
        self._modules: dict[str, ModuleSpec[Any, Any]] = {}
        self._names: dict[str, str] = {}

    def register(
        self,
        spec_or_name: ModuleSpec[Any, Any] | str,
        scene_class: type[Any] | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if isinstance(spec_or_name, ModuleSpec):
            spec = spec_or_name
        else:
            if scene_class is None:
                raise RegistryError("A scene class is required for legacy registration.")
            spec = ModuleSpec(
                key=spec_or_name,
                name=spec_or_name,
                description=f"Legacy scene {spec_or_name}",
                input_model=dict,
                engine=None,
                scene_factory=lambda: scene_class,
            )

        if not spec.key:
            raise RegistryError("Module key cannot be empty.")
        if spec.key in self._modules and not replace:
            raise RegistryError(f"Module '{spec.key}' is already registered.")
        self._modules[spec.key] = spec
        self._names[spec.name] = spec.key

    def get(self, key: str) -> ModuleSpec[Any, Any]:
        if key in self._modules:
            return self._modules[key]
        if key in self._names:
            return self._modules[self._names[key]]
        raise RegistryError(f"Module '{key}' is not registered.")

    def all(self) -> tuple[ModuleSpec[Any, Any], ...]:
        return tuple(self._modules.values())

    def clear(self) -> None:
        self._modules.clear()
        self._names.clear()

    def get_scene(self, name: str) -> type[Any]:
        spec = self.get(name)
        if spec.scene_factory is None:
            raise RegistryError(f"Module '{name}' has no scene factory.")
        scene_class = spec.scene_factory()
        if not isinstance(scene_class, type):
            raise RegistryError(f"Module '{name}' scene factory did not return a class.")
        return scene_class

    def get_all_registered_names(self) -> list[str]:
        return [module.name for module in self.all()]


registry: SimulationRegistry[Any, Any] = SimulationRegistry()
