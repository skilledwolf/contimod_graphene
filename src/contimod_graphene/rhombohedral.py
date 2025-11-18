from contimod.hamiltonian import ContinuumHamiltonian
from contimod.utils import pauli, paulikron, matrix_basis
import numpy as np
import scipy as sp
import jax; jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from contimod_graphene.utils import extract_params, layer_coordinates, sublattice_coordinates, construct_ll_ops
from contimod_graphene.params import *

##############################################################################
# General Multilayer Graphene
##############################################################################

def get_hamiltonian(N_layers=2, params=graphene_params_BLG):
    """
    Get the Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        N_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters.

    Returns:
        function: A JIT-compiled function `h(kx, ky)` that returns the Hamiltonian matrix.
    """
    h_func = partial(hamiltonian, N_layers=N_layers, params=params)
    return h_func

def get_2band_hamiltonian(N_layers=2, params=graphene_params_BLG):
    """
    Get the effective 2-band Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        N_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters.

    Returns:
        function: A JIT-compiled function `h(kx, ky)` that returns the 2x2 effective Hamiltonian matrix.
    """
    h_func = partial(hamiltonian_2bands, N_layers=N_layers, params=params)
    return h_func

def get_hamiltonian_LL(N_layers=2, Ncut=50, flip_valley=False, params=graphene_params_BLG):
    """
    Get the Landau Level Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        N_layers (int): Number of layers.
        Ncut (int): Cutoff for the number of Landau levels.
        flip_valley (bool): If True, returns the Hamiltonian for the K' valley. Default is False (K valley).
        params (dict): Dictionary of graphene parameters.

    Returns:
        function: A function `h(B)` that returns the Hamiltonian matrix for a given magnetic field B.
    """
    h_func = partial(hamiltonian_LL, N_layers=N_layers, Ncut=Ncut, flip_valley=flip_valley, params=params)
    return h_func

##############################################################################
# General Multilayer Graphene
##############################################################################

@partial(jax.jit, static_argnames=['N_layers'])
def hamiltonian(kx, ky, N_layers=3, params=graphene_params_BLG):
    """
    Construct the zero-field Hamiltonian for N-layer Rhombohedral (ABC) graphene.

    Args:
        kx (float): Momentum in x-direction.
        ky (float): Momentum in y-direction.
        N_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters.

    Returns:
        jax.numpy.ndarray: The Hamiltonian matrix of shape (2*N_layers, 2*N_layers).
    """
    keys = ["gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U", "Delta", "delta"]
    gamma0, gamma1, gamma2, gamma3, gamma4, U, Delta, delta = extract_params(params, keys)

    v = jnp.sqrt(3) * gamma0 / 2
    v3 = jnp.sqrt(3) * gamma3 / 2
    v4 = jnp.sqrt(3) * gamma4 / 2
    pi = kx + 1j * ky

    # Use D block for both single-layer and multilayer cases
    D = jnp.array([[0, v * jnp.conj(pi)], 
                    [0, 0]])

    if N_layers < 2:
        return D + D.T.conj()

    # Base Hamiltonian block for a multilayer system
    V = jnp.array([[-v4 * jnp.conj(pi), v3 * pi], 
                    [gamma1, -v4 * jnp.conj(pi)]])
    W = jnp.array([[0, gamma2 / 2], 
                    [0, 0]])

    # Construct Hamiltonian
    M = jnp.kron(jnp.eye(N_layers), D) + jnp.kron(jnp.diag(jnp.ones(N_layers - 1), k=1), V)
    if N_layers > 2:
        M += jnp.kron(jnp.diag(jnp.ones(N_layers - 2), k=2), W)

    M = M + M.conj().T
    M = M + jnp.kron(jnp.diag(U*jnp.linspace(1/ 2, -1/ 2, N_layers)), jnp.eye(2))

    return M

@partial(jax.jit, static_argnames=['N_layers'])
def hamiltonian_2bands(kx, ky, N_layers=3, params=graphene_params_BLG):
    """
    Compute the effective two-band Hamiltonian for an N-layer ABC stacked graphene system
    following the projection method in Eq. (1) of arXiv:0906.4634.

    Parameters:
      k : array-like
          The momentum vector with components k[0] and k[1].
      N : int
          Number of layers in the ABC stack (N must be > 0).
      params : dict
          Dictionary of parameters including:
            "gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U", "Delta", "delta"

    Returns:
      A 2x2 JAX array representing the effective, numerically projected Hamiltonian.
    """
    # Extract tight-binding parameters.
    keys = ["gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U", "Delta", "delta"]
    gamma0, gamma1, gamma2, gamma3, gamma4, U, Delta, delta = extract_params(params, keys)
    
    # Define velocities (in units of ℏ*pi/a)
    v  = jnp.sqrt(3) * gamma0 / 2
    v3 = jnp.sqrt(3) * gamma3 / 2
    v4 = jnp.sqrt(3) * gamma4 / 2

    # Valley index; here we assume xi = 1.0.
    xi = 1.0
    pi = xi * kx + 1j * ky

    # Define the building blocks for the full Hamiltonian.
    D = jnp.array([[0,         v * jnp.conj(pi)],
                   [0,         0           ]])
    V = jnp.array([[-v4 * jnp.conj(pi), v3 * pi],
                   [ gamma1,           -v4 * jnp.conj(pi)]])
    W = jnp.array([[0,                gamma2/2],
                   [0,                0         ]])
    
    # Build the full (2N x 2N) Hamiltonian matrix.
    M = jnp.kron(jnp.eye(N_layers), D) + jnp.kron(jnp.diag(jnp.ones(N_layers - 1), k=1), V)
    if N_layers > 2:
        M = M + jnp.kron(jnp.diag(jnp.ones(N_layers - 2), k=2), W)
    M = M + M.conj().T  # Ensure M is Hermitian.
    
    # Add a potential term if U is nonzero.
    M = M + jnp.kron(jnp.diag(jnp.linspace(U/2, -U/2, N_layers)), jnp.eye(2))
    
    # Project out the internal layers.
    # H11 corresponds to the (1,1) block from the first and last layers.
    H11 = jnp.array([
        [M[0, 0],    M[0, -1]],
        [M[-1, 0],   M[-1, -1]]
    ])
    
    # H22 is the Hamiltonian for the intermediate layers.
    H22 = jnp.array(M[1:-1, 1:-1])
    
    # H12 couples the outer layers to the intermediate ones.
    H12 = jnp.vstack([M[0, 1:-1], M[-1, 1:-1]])
    
    id2 = jnp.eye(2)
    
    # Compute the renormalization matrix S = (id2 + H12 H22^(-2) H12†)^(-1/2).
    S = jnp.linalg.inv(jsp.linalg.sqrtm(id2 + H12 @ jnp.linalg.matrix_power(H22, -2) @ (H12.T.conj())))
    
    # The unrenormalized effective Hamiltonian H0 = H11 - H12 H22^(-1) H12†.
    H0 = H11 - H12 @ jnp.linalg.inv(H22) @ (H12.T.conj())
    
    # Return the renormalized effective 2x2 Hamiltonian.
    return S.T.conj() @ H0 @ S

##############################################################################
# Multilayer Landau Level basis
##############################################################################

    if not flip_valley:
        N_A, N_B = Ncut - 1, Ncut
    else:
        N_A, N_B = Ncut, Ncut - 1

    ops = construct_ll_ops(N_A, N_B)

    # ---------- per-layer blocks ----------
    # K valley uses H_AB ~ v * a; K' uses H_AB ~ -v * a^\dagger
    if not flip_valley:
        # A<-B (intralayer Dirac)
        D_AB = v * rt2_over_lB * ops["a_BA"]          # H_AB ∝ a
        # A<-A and B<-B (interlayer γ4 terms)
        V_AA = -v4 * rt2_over_lB * ops["a_A"]
        V_BB = -v4 * rt2_over_lB * ops["a_B"]
        # A<-B (interlayer γ3 term)
        V_AB =  v3 * rt2_over_lB * ops["adag_BA"]     # ∝ a^\dagger
    else:
        D_AB = -v * rt2_over_lB * ops["adag_BA"]      # H_AB ∝ -a^\dagger
        V_AA =  v4 * rt2_over_lB * ops["adag_A"]
        V_BB =  v4 * rt2_over_lB * ops["adag_B"]
        V_AB = -v3 * rt2_over_lB * ops["a_BA"]

    # Zeros for convenience
    Z_AA = np.zeros((N_A, N_A))
    Z_BB = np.zeros((N_B, N_B))
    Z_BA = np.zeros((N_B, N_A))

    # Intra-layer D block (square of size N_A+N_B). The BA block comes from Hermitian conjugation later.
    D = np.block([[Z_AA, D_AB],
                  [Z_BA, Z_BB]])

    # First nearest interlayer block V (couples layer ℓ to ℓ+1)
    V = np.block([[V_AA,               V_AB],
                  [γ1 * ops["I_BA"],   V_BB]])

    # Second nearest interlayer block W (γ2)
    W = np.block([[np.zeros_like(Z_AA), (γ2 / 2.0) * ops["I_AB"]],
                  [Z_BA,                Z_BB]])

    # ---------- assemble multilayer Hamiltonian ----------
    d_layer = N_A + N_B
    M = np.kron(np.eye(N_layers), D)
    if N_layers > 1:
        M += np.kron(np.diag(np.ones(N_layers - 1), k=1), V)
    if N_layers > 2:
        M += np.kron(np.diag(np.ones(N_layers - 2), k=2), W)

    # Hermitize (adds the BA block and interlayer lower-diagonal blocks)
    M = M + M.conj().T

    # Layer potential (U) distributed linearly across layers, same on both sublattices
    if not np.isclose(U, 0.0):
        M += np.kron(np.diag(np.linspace(U/2.0, -U/2.0, N_layers)), np.eye(d_layer))

    return M
