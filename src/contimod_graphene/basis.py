"""Basis utilities shared across Bernal and rhombohedral models.

All functions are pure NumPy/JAX and avoid dependencies on contimod so
contimod_graphene stays self-contained.
"""

from __future__ import annotations

from numbers import Integral
from typing import Optional, Sequence, Union

import numpy as np

# Local Pauli matrices (kept here to avoid depending on contimod.utils)
PAULI = (
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def _validate_n_layers(n_layers: int) -> int:
    n_layers = int(n_layers)
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    return n_layers


def _normalize_layer_selection(
    n_layers: int,
    layer: int | Sequence[int] | None,
) -> frozenset[int] | None:
    if layer is None:
        return None

    if isinstance(layer, Integral):
        values = (int(layer),)
    else:
        values = tuple(int(entry) for entry in layer)

    if not values:
        raise ValueError("layer selection cannot be empty")
    if any(entry < 1 or entry > n_layers for entry in values):
        raise ValueError(f"layer selections must lie between 1 and {n_layers}")
    return frozenset(values)


def _normalize_sublattice_selection(
    sublattice: str | Sequence[str] | None,
) -> frozenset[str] | None:
    if sublattice is None:
        return None

    values = (sublattice,) if isinstance(sublattice, str) else tuple(sublattice)
    if not values:
        raise ValueError("sublattice selection cannot be empty")

    normalized = []
    for entry in values:
        key = str(entry).strip().upper()
        if key not in {"A", "B"}:
            raise ValueError("sublattice selections must be 'A' and/or 'B'")
        normalized.append(key)
    return frozenset(normalized)


def zero_field_orbital_labels(n_layers: int) -> tuple[str, ...]:
    """Return the zero-field orbital labels `(A1, B1, ..., AN, BN)`."""
    n_layers = _validate_n_layers(n_layers)
    return tuple(
        f"{sublattice}{layer}"
        for layer in range(1, n_layers + 1)
        for sublattice in ("A", "B")
    )


def zero_field_orbital_index(n_layers: int, layer: int, sublattice: str) -> int:
    """Return the zero-field orbital index for a named site."""
    n_layers = _validate_n_layers(n_layers)
    layer = int(layer)
    if layer < 1 or layer > n_layers:
        raise ValueError(f"layer must lie between 1 and {n_layers}")

    sublattice_key = str(sublattice).strip().upper()
    if sublattice_key not in {"A", "B"}:
        raise ValueError("sublattice must be either 'A' or 'B'")
    return 2 * (layer - 1) + (1 if sublattice_key == "B" else 0)


def zero_field_orbital_mask(
    n_layers: int,
    *,
    layer: int | Sequence[int] | None = None,
    sublattice: str | Sequence[str] | None = None,
) -> np.ndarray:
    """Return a boolean mask selecting zero-field orbitals by layer and/or sublattice."""
    n_layers = _validate_n_layers(n_layers)
    layers = _normalize_layer_selection(n_layers, layer)
    sublattices = _normalize_sublattice_selection(sublattice)

    mask = np.ones(2 * n_layers, dtype=bool)
    if layers is not None:
        orbital_layers = np.repeat(np.arange(1, n_layers + 1), 2)
        mask &= np.isin(orbital_layers, tuple(sorted(layers)))
    if sublattices is not None:
        orbital_sublattices = np.tile(np.array(["A", "B"], dtype=object), n_layers)
        mask &= np.isin(orbital_sublattices, tuple(sorted(sublattices)))
    return mask


def zero_field_orbital_projector(
    n_layers: int,
    *,
    layer: int | Sequence[int] | None = None,
    sublattice: str | Sequence[str] | None = None,
    dtype: type[np.complexfloating] | type[np.floating] | type[np.bool_] = complex,
) -> np.ndarray:
    """Return the diagonal projector associated with a zero-field orbital selection."""
    mask = zero_field_orbital_mask(n_layers, layer=layer, sublattice=sublattice)
    return np.diag(mask.astype(dtype))


def bernal_nondimer_mask(n_layers: int) -> np.ndarray:
    """Return a mask selecting the zero-field Bernal non-dimer subspace."""
    n_layers = _validate_n_layers(n_layers)
    mask = np.zeros(2 * n_layers, dtype=bool)
    for layer in range(1, n_layers + 1):
        sublattice = "A" if layer % 2 else "B"
        mask[zero_field_orbital_index(n_layers, layer, sublattice)] = True
    return mask


def bernal_dimer_mask(n_layers: int) -> np.ndarray:
    """Return a mask selecting the zero-field Bernal dimer subspace."""
    return ~bernal_nondimer_mask(n_layers)


def bernal_trilayer_mirror_unitary(
    dtype: type[np.complexfloating] | type[np.floating] | type[np.bool_] = complex,
) -> np.ndarray:
    """Return the ABA-trilayer mirror basis unitary for `(A1, B1, A2, B2, A3, B3)`.

    The returned columns are ordered as:
    `((A1-A3)/sqrt(2), (B1-B3)/sqrt(2), (A1+A3)/sqrt(2), (B1+B3)/sqrt(2), A2, B2)`.
    The first two columns therefore span the odd mirror-parity sector and the last
    four span the even sector.
    """
    scale = 1.0 / np.sqrt(2.0)
    U = np.zeros((6, 6), dtype=dtype)

    a1 = zero_field_orbital_index(3, 1, "A")
    b1 = zero_field_orbital_index(3, 1, "B")
    a2 = zero_field_orbital_index(3, 2, "A")
    b2 = zero_field_orbital_index(3, 2, "B")
    a3 = zero_field_orbital_index(3, 3, "A")
    b3 = zero_field_orbital_index(3, 3, "B")

    U[a1, 0] = scale
    U[a3, 0] = -scale
    U[b1, 1] = scale
    U[b3, 1] = -scale
    U[a1, 2] = scale
    U[a3, 2] = scale
    U[b1, 3] = scale
    U[b3, 3] = scale
    U[a2, 4] = 1.0
    U[b2, 5] = 1.0
    return U


def bernal_trilayer_mirror_projectors(
    dtype: type[np.complexfloating] | type[np.floating] | type[np.bool_] = complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the `(odd, even)` mirror-parity projectors for ABA trilayer graphene."""
    unitary = bernal_trilayer_mirror_unitary(dtype=complex)
    odd = unitary[:, :2] @ unitary[:, :2].conj().T
    even = unitary[:, 2:] @ unitary[:, 2:].conj().T
    return np.asarray(odd, dtype=dtype), np.asarray(even, dtype=dtype)


def rhombohedral_outer_site_indices(n_layers: int) -> tuple[int, int]:
    """Return the zero-field low-energy site indices `(A1, B_N)` for ABC stacks."""
    n_layers = _validate_n_layers(n_layers)
    if n_layers < 2:
        raise ValueError("rhombohedral outer-site indices require n_layers >= 2")
    return (
        zero_field_orbital_index(n_layers, 1, "A"),
        zero_field_orbital_index(n_layers, n_layers, "B"),
    )


def layer_coordinates(n_layers: int) -> np.ndarray:
    """Return per-orbital layer index (A,B per layer)."""
    n_layers = _validate_n_layers(n_layers)
    return np.repeat(np.linspace(-1.0, 1.0, n_layers), 2) if n_layers > 1 else np.array([0.0, 0.0])


def sublattice_coordinates(n_layers: int) -> np.ndarray:
    """Return per-orbital sublattice index (0 for A, 1 for B)."""
    n_layers = _validate_n_layers(n_layers)
    return np.tile([0.0, 1.0], n_layers) if n_layers > 1 else np.array([0.0, 1.0])


def paulikron_local(*factors: Optional[Union[int, np.ndarray]]) -> np.ndarray:
    """Kronecker product skipping None factors (simplified paulikron)."""
    result = None
    for f in factors:
        if f is None:
            continue
        result = f if result is None else np.kron(result, f)
    if result is None:
        raise ValueError("paulikron_local: no factors provided")
    return result


def build_ops(
    matrixdim: int,
    *,
    valleyful: bool,
    spinful: bool,
    layer_vec: np.ndarray,
    sublattice_vec: np.ndarray,
) -> dict:
    """Construct common operators given basis metadata."""
    id2 = np.ones(2)
    ops = {}

    ops["layer"] = paulikron_local(id2 if spinful else None, id2 if valleyful else None, layer_vec)
    ops["sublattice"] = paulikron_local(id2 if spinful else None, id2 if valleyful else None, sublattice_vec)
    ops["identity"] = np.identity(matrixdim * (2 ** valleyful) * (2 ** spinful))

    def valley_op(i: int) -> np.ndarray:
        assert valleyful, "Valley operator not defined in projected subspace."
        return paulikron_local(
            PAULI[0] if spinful else None,
            PAULI[int(i)],
            np.identity(matrixdim),
        )

    def spin_op(i: int) -> np.ndarray:
        assert spinful, "Spin operator not defined in projected subspace."
        return paulikron_local(
            PAULI[int(i)],
            PAULI[0] if valleyful else None,
            np.identity(matrixdim),
        )

    ops["valley_op"] = valley_op
    ops["spin_op"] = spin_op
    return ops


__all__ = [
    "PAULI",
    "bernal_dimer_mask",
    "bernal_nondimer_mask",
    "bernal_trilayer_mirror_projectors",
    "bernal_trilayer_mirror_unitary",
    "build_ops",
    "layer_coordinates",
    "paulikron_local",
    "rhombohedral_outer_site_indices",
    "sublattice_coordinates",
    "zero_field_orbital_index",
    "zero_field_orbital_labels",
    "zero_field_orbital_mask",
    "zero_field_orbital_projector",
]
