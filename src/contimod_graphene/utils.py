import numpy as np
import jax
import jax.numpy as jnp

from .basis import layer_coordinates, sublattice_coordinates

def extract_params(params, keys):
    """
    Extract parameters from a dictionary.

    Args:
        params (dict): Dictionary of parameters.
        keys (list): List of keys to extract.

    Returns:
        list: List of values corresponding to the keys. Returns 0.0 if a key is missing.
    """
    return [params.get(key, 0.0) for key in keys]

def construct_ll_ops(N_A: int, N_B: int):
    """
    Build square (A,A) and (B,B) ladder operators and rectangular (A<-B) and (B<-A)
    maps consistent with the Dirac LL algebra.

    Args:
        N_A (int): Dimension of sublattice A basis.
        N_B (int): Dimension of sublattice B basis.

    Returns:
        dict: Dictionary containing ladder operators and identity matrices.
    """
    # square lowering (superdiagonal) and raising (transpose)
    a_A = np.zeros((N_A, N_A));            a_B = np.zeros((N_B, N_B))
    for n in range(1, N_A): a_A[n-1, n] = np.sqrt(n)
    for n in range(1, N_B): a_B[n-1, n] = np.sqrt(n)
    adag_A = a_A.T.copy();                  adag_B = a_B.T.copy()

    # rectangular A <- B: a (lowering) puts weight on row n-1, col n
    a_BA = np.zeros((N_A, N_B))
    for n in range(1, N_B):
        if n-1 < N_A:
            a_BA[n-1, n] = np.sqrt(n)

    # rectangular A <- B: a^\dagger (raising) puts weight on row n+1, col n
    adag_BA = np.zeros((N_A, N_B))
    for n in range(0, N_B):                 # inclusive upper bound (important!)
        if n+1 < N_A:
            adag_BA[n+1, n] = np.sqrt(n+1)

    # rectangular B <- A: a lowers A-index by 1
    a_AB = np.zeros((N_B, N_A))
    for m in range(1, N_A):
        if m-1 < N_B:
            a_AB[m-1, m] = np.sqrt(m)

    # rectangular B <- A: a^\dagger raises A-index by 1
    adag_AB = np.zeros((N_B, N_A))
    for m in range(0, N_A):                 # inclusive upper bound (important!)
        if m+1 < N_B:
            adag_AB[m+1, m] = np.sqrt(m+1)

    # rectangular index-preserving maps
    I_AB = np.zeros((N_A, N_B))
    for n in range(min(N_A, N_B)):
        I_AB[n, n] = 1.0
    I_BA = I_AB.T.copy()

    return dict(
        a_A=a_A, adag_A=adag_A, a_B=a_B, adag_B=adag_B,
        a_BA=a_BA, adag_BA=adag_BA, a_AB=a_AB, adag_AB=adag_AB,
        I_AB=I_AB, I_BA=I_BA
    )


def batch_hamiltonian(h_fn, *, jit: bool = True):
    """Vectorize a single-k Hamiltonian callable over k-arrays (last dim = 2)."""

    def _batched(ks):
        ks = jnp.asarray(ks)
        orig_shape = ks.shape[:-1]
        ks_flat = ks.reshape((-1, ks.shape[-1]))
        out = jax.vmap(h_fn)(ks_flat)
        return out.reshape(orig_shape + out.shape[1:])

    return jax.jit(_batched) if jit else _batched


__all__ = [
    "batch_hamiltonian",
    "construct_ll_ops",
    "extract_params",
    "layer_coordinates",
    "sublattice_coordinates",
]
