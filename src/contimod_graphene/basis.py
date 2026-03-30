"""Basis utilities shared across Bernal and rhombohedral models.

All functions are pure NumPy/JAX and avoid dependencies on contimod so
contimod_graphene stays self-contained.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

# Local Pauli matrices (kept here to avoid depending on contimod.utils)
PAULI = (
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def layer_coordinates(n_layers: int) -> np.ndarray:
    """Return per-orbital layer index (A,B per layer)."""
    return np.repeat(np.linspace(-1.0, 1.0, n_layers), 2) if n_layers > 1 else np.array([0.0, 0.0])


def sublattice_coordinates(n_layers: int) -> np.ndarray:
    """Return per-orbital sublattice index (0 for A, 1 for B)."""
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
