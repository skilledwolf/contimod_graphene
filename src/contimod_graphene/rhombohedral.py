from contimod.hamiltonian import ContinuumHamiltonian
from contimod.utils import pauli, paulikron, matrix_basis
import numpy as np
import scipy as sp
import jax; jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

# Utility function to extract graphene parameters
def extract_params(params, keys):
    return [params.get(key, 0.0) for key in keys]

from contimod_graphene.params import *


def layer_coordinates(N_layers):
    z_extent = N_layers * 0.335 # nm 
    return np.repeat(np.linspace(-0.5, 0.5, N_layers), 2) if N_layers > 1 else np.array([0.0, 0.0])

def sublattice_coordinates(N_layers):
    return np.tile([0.0, 1.0], N_layers) if N_layers > 1 else np.array([0.0, 1.0])

##############################################################################
# General Multilayer Graphene
##############################################################################

def get_hamiltonian(N_layers=2, params=graphene_params_BLG):
    h_func = partial(hamiltonian, N_layers=N_layers, params=params)
    return h_func

def get_2band_hamiltonian(N_layers=2, params=graphene_params_BLG):
    h_func = partial(hamiltonian_2bands, N_layers=N_layers, params=params)
    return h_func

def get_hamiltonian_LL(N_layers=2, Ncut=50, flip_valley=False, params=graphene_params_BLG):
    h_func = partial(hamiltonian_LL, N_layers=N_layers, Ncut=Ncut, flip_valley=flip_valley, params=params)
    return h_func

##############################################################################
# General Multilayer Graphene
##############################################################################

@partial(jax.jit, static_argnames=['N_layers'])
def hamiltonian(kx, ky, N_layers=3, params=graphene_params_BLG):
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

def hamiltonian_LL(B, N_layers=3, Ncut=50, flip_valley=False, params=graphene_params_BLG):
    # Magnetic length
    # a = 2.46  # Angstrom
    # l_B = 260 / np.sqrt(B)  # magnetic length in Angstrom where B is magnetic field in Tesla
    l_B = 104.29 / np.sqrt(B)  # magnetic length in Angstrom where B is magnetic field in Tesla

    # Define creation and annihilation operators for the Landau levels
    a = np.diag(np.sqrt(np.arange(1, Ncut)), +1)  # Annihilation operator
    a_dag = np.diag(np.sqrt(np.arange(1, Ncut)), -1)  # Creation operator

    # pi and pi^dagger in terms of ladder operators
    pi = np.sqrt(2.0) / l_B * a
    pi_dag = np.sqrt(2.0) / l_B * a_dag

    if flip_valley:
        (pi, pi_dag) = (-pi_dag, -pi)

    # Identity matrices for blocks
    INcut = np.eye(Ncut)  # Identity for the Landau level space

    # Retrieve parameters with defaults
    p = lambda x: params.get(x, 0.0)
    v = np.sqrt(3) * p("gamma0") / 2
    v3 = np.sqrt(3) * p("gamma3") / 2
    v4 = np.sqrt(3) * p("gamma4") / 2
    gamma1 = p("gamma1")
    gamma2 = p("gamma2")
    U = p("U")

    # Define D, V, W matrices using np.block for proper block assembly
    D = np.block([
        [np.zeros((Ncut, Ncut)), v * pi_dag],
        [np.zeros((Ncut, Ncut)), np.zeros((Ncut, Ncut))]
    ])

    V = np.block([
        [-v4 * pi_dag, v3 * pi],
        [gamma1 * INcut, -v4 * pi_dag]
    ])

    W = np.block([
        [np.zeros((Ncut, Ncut)), (gamma2 / 2) * INcut],
        [np.zeros((Ncut, Ncut)), np.zeros((Ncut, Ncut))]
    ])

    # Initialize Hamiltonian M
    M = np.kron(np.eye(N_layers), D)  # Diagonal blocks

    # Add V matrices on the first off-diagonal
    if N_layers > 1:
        M += np.kron(np.diag(np.ones(N_layers - 1), k=1), V)

    # Add W matrices on the second off-diagonal if N > 2
    if N_layers > 2:
        M += np.kron(np.diag(np.ones(N_layers - 2), k=2), W)

    # Ensure the Hamiltonian is Hermitian
    M = M + M.conj().T

    # Add the potential due to the on-site energies (if any)
    if not np.isclose(U, 0.0):
        potential = np.kron(
            np.diag(np.linspace(U / 2, -U / 2, N_layers)),
            np.eye(2 * Ncut)
        )
        M += potential

    return M


