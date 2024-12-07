from contimod.hamiltonian import ContinuumHamiltonian
from contimod.utils import pauli, paulikron, matrix_basis
import numpy as np
import scipy as sp
import jax; jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp

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

def hamiltonian(k, N_layers=3, params=graphene_params_BLG):
    keys = ["gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U", "Delta", "delta"]
    gamma0, gamma1, gamma2, gamma3, gamma4, U, Delta, delta = extract_params(params, keys)

    v = np.sqrt(3) * gamma0 / 2
    v3 = np.sqrt(3) * gamma3 / 2
    v4 = np.sqrt(3) * gamma4 / 2
    pi = k[0] + 1j * k[1]

    # Use D block for both single-layer and multilayer cases
    D = jnp.array([[0, v * jnp.conj(pi)], 
                    [0, 0]])

    if N_layers == 1:
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

    # Add potential due to on-site energies if needed
    if not np.isclose(U, 0.0):
        M = M + jnp.kron(jnp.diag(np.linspace(U / 2, -U / 2, N_layers)), jnp.eye(2))

    return M

##############################################################################
# Multilayer Landau Level basis
##############################################################################

def hamiltonian_LL(B, N_layers=3, Ncut=50, flip_valley=False, params=graphene_params_BLG):
    # Magnetic length
    a = 2.46  # Angstrom
    l_B = 260 / np.sqrt(B)  # magnetic length in Angstrom where B is magnetic field in Tesla

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


