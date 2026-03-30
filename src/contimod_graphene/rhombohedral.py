import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from contimod_graphene.utils import extract_params, layer_coordinates, sublattice_coordinates, construct_ll_ops
from contimod_graphene.params import graphene_params_TLG

##############################################################################
# General Rhombohedral / ABC Multilayer Graphene
##############################################################################

@partial(jax.jit, static_argnames=['n_layers'])
def hamiltonian(kx: float, ky: float, n_layers: int = 3, params: dict = graphene_params_TLG) -> jnp.ndarray:
    """
    Construct the zero-field Hamiltonian for N-layer Rhombohedral (ABC) graphene.

    Args:
        kx (float): Momentum in x-direction.
        ky (float): Momentum in y-direction.
        n_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters.

    Returns:
        jax.numpy.ndarray: The Hamiltonian matrix of shape (2*n_layers, 2*n_layers).
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

    if n_layers < 2:
        return D + D.T.conj()

    # Base Hamiltonian block for a multilayer system
    V = jnp.array([[-v4 * jnp.conj(pi), v3 * pi], 
                    [gamma1, -v4 * jnp.conj(pi)]])
    W = jnp.array([[0, gamma2 / 2], 
                    [0, 0]])

    # Construct Hamiltonian
    M = jnp.kron(jnp.eye(n_layers), D) + jnp.kron(jnp.diag(jnp.ones(n_layers - 1), k=1), V)
    if n_layers > 2:
        M += jnp.kron(jnp.diag(jnp.ones(n_layers - 2), k=2), W)

    M = M + M.conj().T
    M = M + jnp.kron(jnp.diag(U*jnp.linspace(1/ 2, -1/ 2, n_layers)), jnp.eye(2))

    return M

@partial(jax.jit, static_argnames=['n_layers'])
def hamiltonian_2bands(kx: float, ky: float, n_layers: int = 3, params: dict = graphene_params_TLG) -> jnp.ndarray:
    """
    Compute the effective two-band Hamiltonian for an N-layer ABC stacked graphene system
    following the projection method in Eq. (1) of arXiv:0906.4634.

    Parameters:
      kx (float): Momentum in x-direction.
      ky (float): Momentum in y-direction.
      n_layers (int): Number of layers in the ABC stack (N must be > 0).
      params (dict): Dictionary of parameters including:
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
    M = jnp.kron(jnp.eye(n_layers), D) + jnp.kron(jnp.diag(jnp.ones(n_layers - 1), k=1), V)
    if n_layers > 2:
        M = M + jnp.kron(jnp.diag(jnp.ones(n_layers - 2), k=2), W)
    M = M + M.conj().T  # Ensure M is Hermitian.
    
    # Add a potential term if U is nonzero.
    M = M + jnp.kron(jnp.diag(jnp.linspace(U/2, -U/2, n_layers)), jnp.eye(2))
    
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

def hamiltonian_LL(B: float, n_layers: int = 3, n_cut: int = 50, flip_valley: bool = False, params: dict = graphene_params_TLG) -> np.ndarray:
    """
    Multilayer (ABC) graphene Landau-level Hamiltonian in an asymmetric LL basis
    that removes unphysical LLs by using different numbers of LLs on the two
    sublattices. For valley K we use (N_B, N_A) = (n_cut, n_cut-1); for K' we swap.

    Args:
      B: magnetic field [T]
      n_layers: number of layers
      n_cut: LL cutoff on the sublattice that hosts the n=0 mode
      flip_valley: if True, build K' (swap sublattices + sign switches)
      params: dict with keys "gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U"
    Returns:
      Dense numpy array of shape (n_layers*(2*n_cut-1), n_layers*(2*n_cut-1))
    """

    if n_cut < 2:
        raise ValueError("n_cut must be >= 2 for a meaningful asymmetric LL basis.")

    # magnetic length [Å]
    l_B = 104.29 / np.sqrt(B)
    rt2_over_lB = np.sqrt(2.0) / l_B

    # parameters
    p = lambda x: params.get(x, 0.0)
    v   = np.sqrt(3) * p("gamma0") / 2
    v3  = np.sqrt(3) * p("gamma3") / 2
    v4  = np.sqrt(3) * p("gamma4") / 2
    γ1  = p("gamma1")
    γ2  = p("gamma2")
    U   = p("U")

    # Choose LL dimensions per valley:
    # K valley: zero mode on B    -> (N_B, N_A) = (n_cut,   n_cut-1)
    # K' valley: zero mode on A   -> (N_B, N_A) = (n_cut-1, n_cut)
    if not flip_valley:
        N_A, N_B = n_cut - 1, n_cut
    else:
        N_A, N_B = n_cut, n_cut - 1

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
    M = np.kron(np.eye(n_layers), D)
    if n_layers > 1:
        M += np.kron(np.diag(np.ones(n_layers - 1), k=1), V)
    if n_layers > 2:
        M += np.kron(np.diag(np.ones(n_layers - 2), k=2), W)

    # Hermitize (adds the BA block and interlayer lower-diagonal blocks)
    M = M + M.conj().T

    # Layer potential (U) distributed linearly across layers, same on both sublattices
    if not np.isclose(U, 0.0):
        M += np.kron(np.diag(np.linspace(U/2.0, -U/2.0, n_layers)), np.eye(d_layer))

    return M

def get_hamiltonian(n_layers: int = 3, params: dict = graphene_params_TLG):
    """
    Get the Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        n_layers (int): Number of layers. Defaults to the trilayer ABC entry point.
        params (dict): Dictionary of graphene parameters. Defaults to the ABC/TLG preset.

    Returns:
        function: A JIT-compiled function `h(kx, ky)` that returns the Hamiltonian matrix.
    """
    h_func = partial(hamiltonian, n_layers=n_layers, params=params)
    return h_func

def get_2band_hamiltonian(n_layers: int = 3, params: dict = graphene_params_TLG):
    """
    Get the effective 2-band Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        n_layers (int): Number of layers. Defaults to the trilayer ABC entry point.
        params (dict): Dictionary of graphene parameters. Defaults to the ABC/TLG preset.

    Returns:
        function: A JIT-compiled function `h(kx, ky)` that returns the 2x2 effective Hamiltonian matrix.
    """
    h_func = partial(hamiltonian_2bands, n_layers=n_layers, params=params)
    return h_func

def get_hamiltonian_LL(n_layers: int = 3, n_cut: int = 50, flip_valley: bool = False, params: dict = graphene_params_TLG):
    """
    Get the Landau Level Hamiltonian function for N-layer Rhombohedral (ABC) graphene.

    Args:
        n_layers (int): Number of layers. Defaults to the trilayer ABC entry point.
        n_cut (int): Cutoff for the number of Landau levels.
        flip_valley (bool): If True, returns the Hamiltonian for the K' valley. Default is False (K valley).
        params (dict): Dictionary of graphene parameters. Defaults to the ABC/TLG preset.

    Returns:
        function: A function `h(B)` that returns the Hamiltonian matrix for a given magnetic field B.
    """
    h_func = partial(hamiltonian_LL, n_layers=n_layers, n_cut=n_cut, flip_valley=flip_valley, params=params)
    return h_func
