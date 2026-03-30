"""User-facing multilayer graphene model objects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Mapping

from . import bernal, rhombohedral
from .params import GrapheneTBParameters, load_parameter_set
from .utils import batch_hamiltonian


ParameterInput = str | Path | Mapping[str, Any] | GrapheneTBParameters


def _coerce_parameters(
    params: ParameterInput | None,
    *,
    default_preset: str,
    family: str,
) -> GrapheneTBParameters:
    resolved = load_parameter_set(default_preset) if params is None else load_parameter_set(params)
    return resolved.validate_for(family)


def _flip_valley_from_label(valley: str) -> bool:
    label = valley.strip().upper()
    if label in {"K", "+K", "+", "KP"}:
        return False
    if label in {"K'", "K’", "-K", "-", "KM"}:
        return True
    raise ValueError("valley must be one of 'K', \"K'\", '+', or '-'")


@dataclass(frozen=True)
class BernalMultilayer:
    """Thin wrapper around the low-level Bernal (ABA/ABAB/...) kernels."""

    n_layers: int = 2
    params: ParameterInput | None = field(default=None)

    family: str = field(init=False, default="bernal")
    stacking_label: str = field(init=False, default="Bernal")
    default_preset_name: str = field(init=False, default="blg")

    def __post_init__(self) -> None:
        if int(self.n_layers) < 1:
            raise ValueError("n_layers must be >= 1")
        object.__setattr__(self, "n_layers", int(self.n_layers))
        object.__setattr__(
            self,
            "params",
            _coerce_parameters(self.params, default_preset=self.default_preset_name, family=self.family),
        )

    def replace(self, **changes: Any) -> "BernalMultilayer":
        return dataclass_replace(self, **changes)

    def with_params(self, **overrides: Any) -> "BernalMultilayer":
        return self.replace(params=self.params.replace(**overrides))

    def hamiltonian(self, kx: float, ky: float):
        return bernal.hamiltonian(kx, ky, n_layers=self.n_layers, params=self.params)

    def hamiltonian_batch(self, ks, *, jit: bool = True):
        return batch_hamiltonian(lambda k: self.hamiltonian(k[0], k[1]), jit=jit)(ks)

    def landau_level_hamiltonian(self, B: float, *, n_cut: int, valley: str = "K"):
        return bernal.hamiltonian_LL(
            B,
            n_layers=self.n_layers,
            n_cut=n_cut,
            flip_valley=_flip_valley_from_label(valley),
            params=self.params,
        )

    def two_band_hamiltonian(self, kx: float, ky: float):
        if self.n_layers != 2:
            raise ValueError("BernalMultilayer.two_band_hamiltonian is only defined for bilayer graphene")
        return bernal.hamiltonian_2bands(kx, ky, params=self.params)


@dataclass(frozen=True)
class RhombohedralMultilayer:
    """Thin wrapper around the low-level rhombohedral (ABC...) kernels."""

    n_layers: int = 3
    params: ParameterInput | None = field(default=None)

    family: str = field(init=False, default="rhombohedral")
    stacking_label: str = field(init=False, default="Rhombohedral")
    default_preset_name: str = field(init=False, default="tlg")

    def __post_init__(self) -> None:
        if int(self.n_layers) < 1:
            raise ValueError("n_layers must be >= 1")
        object.__setattr__(self, "n_layers", int(self.n_layers))
        object.__setattr__(
            self,
            "params",
            _coerce_parameters(self.params, default_preset=self.default_preset_name, family=self.family),
        )

    def replace(self, **changes: Any) -> "RhombohedralMultilayer":
        return dataclass_replace(self, **changes)

    def with_params(self, **overrides: Any) -> "RhombohedralMultilayer":
        return self.replace(params=self.params.replace(**overrides))

    def hamiltonian(self, kx: float, ky: float):
        return rhombohedral.hamiltonian(kx, ky, n_layers=self.n_layers, params=self.params)

    def hamiltonian_batch(self, ks, *, jit: bool = True):
        return batch_hamiltonian(lambda k: self.hamiltonian(k[0], k[1]), jit=jit)(ks)

    def landau_level_hamiltonian(self, B: float, *, n_cut: int, valley: str = "K"):
        return rhombohedral.hamiltonian_LL(
            B,
            n_layers=self.n_layers,
            n_cut=n_cut,
            flip_valley=_flip_valley_from_label(valley),
            params=self.params,
        )

    def two_band_hamiltonian(self, kx: float, ky: float):
        if self.n_layers < 2:
            raise ValueError(
                "RhombohedralMultilayer.two_band_hamiltonian requires n_layers >= 2"
            )
        return rhombohedral.hamiltonian_2bands(kx, ky, n_layers=self.n_layers, params=self.params)


ABAMultilayer = BernalMultilayer
ABCMultilayer = RhombohedralMultilayer


__all__ = [
    "ABAMultilayer",
    "ABCMultilayer",
    "BernalMultilayer",
    "RhombohedralMultilayer",
]
