"""Public parameter management for graphene tight-binding models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import jax


_DEFAULT_JSON = files("contimod_graphene.data").joinpath("params.json")

with _DEFAULT_JSON.open("r", encoding="utf-8") as f:
    _DATA = json.load(f)

_ALIASES = {**_DATA.get("aliases", {})}
_KNOWN_PARAMETER_KEYS = (
    "gamma0",
    "gamma1",
    "gamma2",
    "gamma3",
    "gamma4",
    "U",
    "Delta",
    "delta",
    "gamma5",
)
_REQUIRED_PARAMETER_KEYS = (
    "gamma0",
    "gamma1",
    "gamma2",
    "gamma3",
    "gamma4",
    "U",
    "Delta",
    "delta",
)


def _resolve_kind(kind: str) -> str:
    key = kind.lower()
    if key in _DATA["sets"]:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"Unknown graphene parameter set '{kind}'")


def _split_parameter_payload(
    data: Mapping[str, Any],
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
    values = {}
    extras = {}
    present_known_keys = set()

    for key in _KNOWN_PARAMETER_KEYS:
        if key in data:
            values[key] = data[key]
            present_known_keys.add(key)
        elif allow_partial:
            values[key] = 0.0

    missing = [key for key in _REQUIRED_PARAMETER_KEYS if key not in data]
    if missing and not allow_partial:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required graphene parameters: {missing_str}")

    if allow_partial:
        for key in _KNOWN_PARAMETER_KEYS:
            values.setdefault(key, 0.0)

    for key, value in data.items():
        if key not in _KNOWN_PARAMETER_KEYS:
            extras[key] = value

    return values, extras, frozenset(present_known_keys)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GrapheneTBParameters(Mapping[str, Any]):
    """Immutable, mapping-compatible graphene tight-binding parameters."""

    gamma0: Any
    gamma1: Any
    gamma2: Any
    gamma3: Any
    gamma4: Any
    U: Any
    Delta: Any
    delta: Any
    gamma5: Any = 0.0
    extras: Mapping[str, Any] = field(default_factory=dict)
    preset_name: str | None = None
    source: str | None = None
    _present_keys: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))
        if self._present_keys:
            present = frozenset(self._present_keys)
        else:
            present = frozenset(_REQUIRED_PARAMETER_KEYS)
            if "gamma5" in self.extras or self.gamma5 != 0.0:
                present = present | {"gamma5"}
        object.__setattr__(self, "_present_keys", present)

    def __iter__(self):
        yield from self.to_dict().keys()

    def __len__(self) -> int:
        return len(self.to_dict())

    def __getitem__(self, key: str) -> Any:
        if key in _KNOWN_PARAMETER_KEYS:
            if key == "gamma5" and key not in self._present_keys:
                raise KeyError(key)
            return getattr(self, key)
        return self.extras[key]

    def tree_flatten(self):
        extra_keys = tuple(sorted(self.extras))
        children = [getattr(self, key) for key in _KNOWN_PARAMETER_KEYS]
        children.extend(self.extras[key] for key in extra_keys)
        aux = (extra_keys, self.preset_name, self.source, tuple(sorted(self._present_keys)))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        extra_keys, preset_name, source, present_keys = aux_data
        base_count = len(_KNOWN_PARAMETER_KEYS)
        base_values = dict(zip(_KNOWN_PARAMETER_KEYS, children[:base_count]))
        extras = dict(zip(extra_keys, children[base_count:]))
        return cls(
            **base_values,
            extras=extras,
            preset_name=preset_name,
            source=source,
            _present_keys=frozenset(present_keys),
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        preset_name: str | None = None,
        source: str | None = None,
        allow_partial: bool = False,
    ) -> "GrapheneTBParameters":
        values, extras, present_keys = _split_parameter_payload(data, allow_partial=allow_partial)
        return cls(
            **values,
            extras=extras,
            preset_name=preset_name,
            source=source,
            _present_keys=present_keys,
        )

    @classmethod
    def preset(cls, kind: str) -> "GrapheneTBParameters":
        key = _resolve_kind(kind)
        return cls.from_dict(_DATA["sets"][key], preset_name=key, source=str(_DEFAULT_JSON))

    @classmethod
    def from_json(cls, path: str | Path) -> "GrapheneTBParameters":
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)
        return cls.from_dict(data, source=str(path))

    def to_dict(self) -> dict[str, Any]:
        out = {key: getattr(self, key) for key in _REQUIRED_PARAMETER_KEYS}
        if "gamma5" in self._present_keys:
            out["gamma5"] = self.gamma5
        out.update(self.extras)
        return out

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def replace(self, **overrides: Any) -> "GrapheneTBParameters":
        payload = self.to_dict()
        payload.update(overrides)
        return type(self).from_dict(
            payload,
            preset_name=self.preset_name,
            source=self.source,
            allow_partial=False,
        )

    def validate_for(self, family: str) -> "GrapheneTBParameters":
        family_key = family.lower()
        if family_key not in {"bernal", "rhombohedral"}:
            raise ValueError(f"Unknown graphene family '{family}'")
        missing = [key for key in _REQUIRED_PARAMETER_KEYS if key not in self]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Parameters for the {family_key} family are missing required keys: {missing_str}"
            )
        return self


def load_parameter_set(
    name_or_path: str | Path | Mapping[str, Any] | GrapheneTBParameters,
) -> GrapheneTBParameters:
    """Load a validated parameter set from a preset name, path, mapping, or object."""
    if isinstance(name_or_path, GrapheneTBParameters):
        return name_or_path
    if isinstance(name_or_path, Mapping):
        return GrapheneTBParameters.from_dict(name_or_path)

    candidate = Path(name_or_path)
    if candidate.exists():
        return GrapheneTBParameters.from_json(candidate)

    return GrapheneTBParameters.preset(str(name_or_path))


def list_parameter_sets() -> list[str]:
    """Return the canonical built-in parameter-set names."""
    return list(_DATA["sets"].keys())


def get_params(kind: str | Mapping[str, Any] | GrapheneTBParameters) -> GrapheneTBParameters:
    """Compatibility alias for loading a parameter set."""
    return load_parameter_set(kind)


def load(path: str | Path) -> GrapheneTBParameters:
    """Compatibility alias for loading a parameter object from JSON."""
    return GrapheneTBParameters.from_json(path)


def list_sets() -> list[str]:
    """Compatibility alias for listing canonical built-in parameter sets."""
    return list_parameter_sets()


graphene_params = GrapheneTBParameters.preset("slg")
graphene_params_BLG = GrapheneTBParameters.preset("blg")
graphene_params_TLG = GrapheneTBParameters.preset("tlg")
graphene_params_4LG = GrapheneTBParameters.preset("4lg")


__all__ = [
    "GrapheneTBParameters",
    "get_params",
    "graphene_params",
    "graphene_params_4LG",
    "graphene_params_BLG",
    "graphene_params_TLG",
    "list_parameter_sets",
    "list_sets",
    "load",
    "load_parameter_set",
]
