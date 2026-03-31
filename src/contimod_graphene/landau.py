"""Landau-level form-factor helpers for graphene LL workflows."""

from __future__ import annotations

import numpy as np
from scipy.special import eval_genlaguerre, poch


def ll_formfactor(
    n_prime: np.ndarray | int,
    n: np.ndarray | int,
    qx: np.ndarray | float,
    qy: np.ndarray | float = 0,
    a_L: float = 1,
) -> np.ndarray:
    """Return the orbital LL form factor.

    This is the standard single-component Landau-level form factor from
    J. Phys. C 18 (1985) 1003, kept as a pure NumPy/SciPy helper.
    """

    n_min = np.where(n_prime + 1 > n, n, n_prime)
    n_max = np.where(n_prime + 1 > n, n_prime, n)

    n_diff = np.abs(n_prime - n)

    q2 = (qx**2 + qy**2) * a_L**2

    laguerre_value = eval_genlaguerre(n_min, n_diff, 0.5 * q2)

    coeff = np.sqrt(poch(n_max + 1, n_min - n_max))
    exp_term = np.exp(-0.25 * q2)
    power_term = (q2 / 2) ** (n_diff / 2)

    phasearg = np.arctan2(qy, qx) * (n - n_prime) + n_diff * np.pi / 2
    phase = np.cos(phasearg) + 1j * np.sin(phasearg)

    result = phase * coeff * power_term * exp_term * laguerre_value
    return result if np.ndim(result) > 0 else result[()]


def graphene_ll_formfactors(
    wavefunctions: np.ndarray,
    ll_block_sizes: np.ndarray | list[int] | tuple[int, ...],
    qx: np.ndarray | float,
    qy: np.ndarray | float = 0,
    *,
    a_L: float = 1,
) -> np.ndarray:
    """Contract graphene LL eigenvectors with orbital LL form factors.

    Args:
        wavefunctions: Array of shape ``(sum(ll_block_sizes), n_states)``.
        ll_block_sizes: Number of orbital LL basis states carried by each
            graphene orbital block. This supports the asymmetric ``N_A != N_B``
            LL bases used by both Bernal and rhombohedral builders.
        qx: Momentum-transfer x-component.
        qy: Momentum-transfer y-component.
        a_L: Magnetic-length scale used in the orbital form factor.

    Returns:
        Complex array with shape ``broadcast(qx, qy) + (n_states, n_states)``.
    """

    wavefunctions = np.asarray(wavefunctions)
    if wavefunctions.ndim != 2:
        raise ValueError(f"wavefunctions must have shape (n_basis, n_states); got {wavefunctions.shape}.")

    ll_block_sizes = np.asarray(ll_block_sizes, dtype=int)
    if ll_block_sizes.ndim != 1 or np.any(ll_block_sizes <= 0):
        raise ValueError(f"ll_block_sizes must be a 1D positive integer array; got {ll_block_sizes}.")
    if int(np.sum(ll_block_sizes)) != int(wavefunctions.shape[0]):
        raise ValueError(
            f"Wavefunction/basis mismatch: sum(ll_block_sizes)={int(np.sum(ll_block_sizes))} "
            f"but wavefunctions.shape[0]={wavefunctions.shape[0]}."
        )

    qx = np.asarray(qx)
    qy_arr = np.asarray(qy)
    qy_arg = 0 if qy_arr.shape == () and qy_arr == 0 else qy_arr[..., None, None]

    formfactors = None
    offset = 0
    for block_size in ll_block_sizes:
        block_slice = slice(offset, offset + int(block_size))
        ns = np.arange(int(block_size))
        F_block = ll_formfactor(ns[None, :, None], ns[None, None, :], qx[..., None, None], qy=qy_arg, a_L=a_L)
        wf_block = wavefunctions[block_slice, :]
        contrib = np.einsum("in,...ij,jm->...nm", wf_block.conj(), F_block, wf_block, optimize=True)
        formfactors = contrib if formfactors is None else formfactors + contrib
        offset += int(block_size)

    return formfactors


__all__ = ["graphene_ll_formfactors", "ll_formfactor"]
