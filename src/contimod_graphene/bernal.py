import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from contimod_graphene.params import graphene_params_BLG
from contimod_graphene.utils import extract_params, layer_coordinates, sublattice_coordinates, construct_ll_ops

##############################################################################
# General Multilayer Graphene (Bernal / ABA)
##############################################################################

@partial(jax.jit, static_argnames=['n_layers'])
def hamiltonian(kx: float, ky: float, n_layers: int = 3, params: dict = graphene_params_BLG) -> jnp.ndarray: # see 10.1103/PhysRevB.83.165443 or https://ar5iv.labs.arxiv.org/html/1404.1603
    """
    Construct the zero-field Hamiltonian for N-layer Bernal (ABA) graphene.
    
    The Hamiltonian includes:
    - Intralayer hopping (gamma0)
    - Nearest-neighbor interlayer hopping (gamma1, v3, v4)
    - Next-nearest-neighbor interlayer hopping (gamma2, gamma5)
    - On-site potential difference between dimer and non-dimer sites (delta)
    - Interlayer potential difference (U)

    Args:
        kx (float): Momentum in x-direction.
        ky (float): Momentum in y-direction.
        n_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters.

    Returns:
        jax.numpy.ndarray: The Hamiltonian matrix of shape (2*n_layers, 2*n_layers).
    """
    keys = ["gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "gamma5", "U", "Delta", "delta"]
    gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, U, Delta, delta = extract_params(params, keys)

    v = jnp.sqrt(3) * gamma0 / 2
    v3 = jnp.sqrt(3) * gamma3 / 2
    v4 = jnp.sqrt(3) * gamma4 / 2
    pi = kx + 1j * ky

    # Intralayer block D (upper triangle)
    # D + D.T.conj() = [[0, v pi*], [v pi, 0]]
    D = jnp.array([[0, v * jnp.conj(pi)], 
                    [0, 0]])

    if n_layers < 2:
        return D + D.T.conj()

    # Interlayer blocks
    # V_odd (1->2, 3->4...): AB stacking
    # A_j -> B_{j+1} (v3), B_j -> A_{j+1} (gamma1)
    V_odd = jnp.array([[-v4 * jnp.conj(pi), v3 * pi], 
                       [gamma1, -v4 * jnp.conj(pi)]])
    
    # V_even (2->3, 4->5...): BA stacking
    # A_j -> B_{j+1} (gamma1), B_j -> A_{j+1} (v3)
    V_even = jnp.array([[-v4 * pi, gamma1],
                        [v3 * jnp.conj(pi), -v4 * pi]])

    # Next-nearest layer block W (1->3, 2->4...)
    # A_j -> A_{j+2} (gamma2), B_j -> B_{j+2} (gamma5)
    W = jnp.array([[gamma2 / 2, 0], 
                   [0, gamma5 / 2]])

    # Construct Hamiltonian blocks
    # We construct the upper triangular part and then hermitianize
    
    # Initialize with zeros
    # Shape: (2*n_layers, 2*n_layers)
    # But we need to construct it as a JAX array.
    # We can use jnp.block with a list of lists.
    
    blocks = [[jnp.zeros((2,2)) for _ in range(n_layers)] for _ in range(n_layers)]
    
    for i in range(n_layers):
        # Diagonal D
        blocks[i][i] = D
        
        # Superdiagonal V
        if i + 1 < n_layers:
            if i % 2 == 0: # 0->1 (Layer 1->2) -> V_odd
                blocks[i][i+1] = V_odd
            else: # 1->2 (Layer 2->3) -> V_even
                blocks[i][i+1] = V_even
                
        # Next-nearest W
        if i + 2 < n_layers:
            blocks[i][i+2] = W
            
    M = jnp.block(blocks)
    M = M + M.conj().T
    
    # Add on-site potentials (delta)
    # Odd layers (0, 2...): A (non-dimer) -> 0, B (dimer) -> delta
    # Even layers (1, 3...): A (dimer) -> delta, B (non-dimer) -> 0
    
    delta_odd = jnp.diag(jnp.array([0.0, delta]))
    delta_even = jnp.diag(jnp.array([delta, 0.0]))
    
    diags = []
    for i in range(n_layers):
        if i % 2 == 0:
            diags.append(delta_odd)
        else:
            diags.append(delta_even)
            
    M_delta = jax.scipy.linalg.block_diag(*diags)
    M = M + M_delta
    
    # Add U potential
    M = M + jnp.kron(jnp.diag(U*jnp.linspace(1/ 2, -1/ 2, n_layers)), jnp.eye(2))

    return M

def hamiltonian_LL(B: float, n_layers: int = 3, n_cut: int = 50, flip_valley: bool = False, params: dict = graphene_params_BLG) -> np.ndarray:
    """
    Multilayer (ABA) graphene Landau-level Hamiltonian.
    
    Constructs the Hamiltonian in a basis of Landau levels. Uses an asymmetric basis
    (N_A != N_B) to avoid fermion doubling and properly describe the zero-energy modes.

    Args:
        B (float): Magnetic field in Tesla.
        n_layers (int): Number of layers.
        n_cut (int): Cutoff for the number of Landau levels.
        flip_valley (bool): If True, compute for K' valley. Default False (K valley).
        params (dict): Dictionary of graphene parameters.

    Returns:
        numpy.ndarray: The Hamiltonian matrix.
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
    γ5  = p("gamma5")
    U   = p("U")
    delta = p("delta")

    # Choose LL dimensions per valley:
    if not flip_valley:
        N_A, N_B = n_cut - 1, n_cut
    else:
        N_A, N_B = n_cut, n_cut - 1

    ops = construct_ll_ops(N_A, N_B)

    # Zeros for convenience
    Z_AA = np.zeros((N_A, N_A))
    Z_BB = np.zeros((N_B, N_B))
    Z_BA = np.zeros((N_B, N_A))
    Z_AB = np.zeros((N_A, N_B))

    # ---------- per-layer blocks ----------
    
    # D block (Intralayer)
    if not flip_valley:
        # K valley: H_AB ~ v * a
        D_AB = v * rt2_over_lB * ops["a_BA"]
        D = np.block([[Z_AA, D_AB],
                      [Z_BA, Z_BB]])
    else:
        # K' valley: H_AB ~ -v * a^\dagger
        D_AB = -v * rt2_over_lB * ops["adag_BA"]
        D = np.block([[Z_AA, D_AB],
                      [Z_BA, Z_BB]])
                      
    # V_odd (1->2, 3->4...): AB stacking
    if not flip_valley:
        # K valley
        # V_AA = -v4 * a
        V_odd_AA = -v4 * rt2_over_lB * ops["a_A"]
        # V_BB = -v4 * a
        V_odd_BB = -v4 * rt2_over_lB * ops["a_B"]
        # V_AB = v3 * a^\dagger
        V_odd_AB = v3 * rt2_over_lB * ops["adag_BA"]
        # V_BA = gamma1
        V_odd_BA = γ1 * ops["I_BA"]
        
        V_odd = np.block([[V_odd_AA, V_odd_AB],
                          [V_odd_BA, V_odd_BB]])
    else:
        # K' valley
        # V_AA = v4 * a^\dagger
        V_odd_AA = v4 * rt2_over_lB * ops["adag_A"]
        # V_BB = v4 * a^\dagger
        V_odd_BB = v4 * rt2_over_lB * ops["adag_B"]
        # V_AB = -v3 * a
        V_odd_AB = -v3 * rt2_over_lB * ops["a_BA"]
        # V_BA = gamma1
        V_odd_BA = γ1 * ops["I_BA"]
        
        V_odd = np.block([[V_odd_AA, V_odd_AB],
                          [V_odd_BA, V_odd_BB]])
                          
    # V_even (2->3, 4->5...): BA stacking
    if not flip_valley:
        # K valley
        # V_AA = -v4 * a^\dagger
        V_even_AA = -v4 * rt2_over_lB * ops["adag_A"]
        # V_BB = -v4 * a^\dagger
        V_even_BB = -v4 * rt2_over_lB * ops["adag_B"]
        # V_AB = gamma1
        V_even_AB = γ1 * ops["I_AB"]
        # V_BA = v3 * a
        V_even_BA = v3 * rt2_over_lB * ops["a_AB"]
        
        V_even = np.block([[V_even_AA, V_even_AB],
                           [V_even_BA, V_even_BB]])
    else:
        # K' valley
        # V_AA = v4 * a
        V_even_AA = v4 * rt2_over_lB * ops["a_A"]
        # V_BB = v4 * a
        V_even_BB = v4 * rt2_over_lB * ops["a_B"]
        # V_AB = gamma1
        V_even_AB = γ1 * ops["I_AB"]
        # V_BA = -v3 * a^\dagger
        V_even_BA = -v3 * rt2_over_lB * ops["adag_AB"]
        
        V_even = np.block([[V_even_AA, V_even_AB],
                           [V_even_BA, V_even_BB]])

    # W (1->3, 2->4...): Next-nearest
    # W_AA = gamma2/2, W_BB = gamma5/2
    W_AA = (γ2 / 2.0) * np.eye(N_A)
    W_BB = (γ5 / 2.0) * np.eye(N_B)
    W = np.block([[W_AA, Z_AB],
                  [Z_BA, W_BB]])

    # ---------- assemble multilayer Hamiltonian ----------
    d_layer = N_A + N_B
    
    blocks = [[np.zeros((d_layer, d_layer)) for _ in range(n_layers)] for _ in range(n_layers)]
    
    for i in range(n_layers):
        # Diagonal D (upper part only for hermitian sum? No, D is already partial)
        # In rhombohedral.py, D was full block?
        # "D = np.block([[Z_AA, D_AB], [Z_BA, Z_BB]])" -> This is upper triangle.
        # "M = M + M.conj().T" -> This makes it full.
        blocks[i][i] = D
        
        if i + 1 < n_layers:
            if i % 2 == 0:
                blocks[i][i+1] = V_odd
            else:
                blocks[i][i+1] = V_even
                
        if i + 2 < n_layers:
            blocks[i][i+2] = W
            
    M = np.block(blocks)
    M = M + M.conj().T
    
    # Add on-site potentials (delta)
    # Odd layers: A->0, B->delta
    delta_odd_AA = np.zeros((N_A, N_A))
    delta_odd_BB = delta * np.eye(N_B)
    delta_odd_block = np.block([[delta_odd_AA, Z_AB],
                                [Z_BA, delta_odd_BB]])
                                
    # Even layers: A->delta, B->0
    delta_even_AA = delta * np.eye(N_A)
    delta_even_BB = np.zeros((N_B, N_B))
    delta_even_block = np.block([[delta_even_AA, Z_AB],
                                 [Z_BA, delta_even_BB]])
                                 
    diags = []
    for i in range(n_layers):
        if i % 2 == 0:
            diags.append(delta_odd_block)
        else:
            diags.append(delta_even_block)
            
    M_delta = jsp.linalg.block_diag(*diags)
    M += M_delta

    # Layer potential (U)
    if not np.isclose(U, 0.0):
        M += np.kron(np.diag(np.linspace(U/2.0, -U/2.0, n_layers)), np.eye(d_layer))

    return M

def get_hamiltonian(n_layers: int = 2, params: dict = graphene_params_BLG):
    """
    Get the Hamiltonian function for N-layer Bernal (ABA) graphene.

    Args:
        n_layers (int): Number of layers.
        params (dict): Dictionary of graphene parameters (gamma0, gamma1, etc.).

    Returns:
        function: A JIT-compiled function `h(kx, ky)` that returns the Hamiltonian matrix.
    """
    h_func = partial(hamiltonian, n_layers=n_layers, params=params)
    return h_func

def get_hamiltonian_LL(n_layers: int = 2, n_cut: int = 50, flip_valley: bool = False, params: dict = graphene_params_BLG):
    """
    Get the Landau Level Hamiltonian function for N-layer Bernal (ABA) graphene.

    Args:
        n_layers (int): Number of layers.
        n_cut (int): Cutoff for the number of Landau levels.
        flip_valley (bool): If True, returns the Hamiltonian for the K' valley. Default is False (K valley).
        params (dict): Dictionary of graphene parameters.

    Returns:
        function: A function `h(B)` that returns the Hamiltonian matrix for a given magnetic field B.
    """
    h_func = partial(hamiltonian_LL, n_layers=n_layers, n_cut=n_cut, flip_valley=flip_valley, params=params)
    return h_func

@partial(jax.jit, static_argnames=[])
def hamiltonian_2bands(kx: float, ky: float, params: dict = graphene_params_BLG) -> jnp.ndarray:
    """
    Compute the effective two-band Hamiltonian for Bilayer Graphene (Bernal).
    See Eq 30 in https://arxiv.org/pdf/1205.6953.pdf

    Args:
        kx (float): Momentum in x-direction.
        ky (float): Momentum in y-direction.
        params (dict): Dictionary of graphene parameters.

    Returns:
        jax.numpy.ndarray: The Hamiltonian matrix of shape (2, 2).
    """
    keys = ["gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "U", "Delta", "delta"]
    gamma0, gamma1, gamma2, gamma3, gamma4, U, Delta, delta = extract_params(params, keys)

    delta_prime = delta
    delta_AB = Delta
    v = jnp.sqrt(3)*gamma0/2  # in units of hbar/a
    v3 = jnp.sqrt(3)*gamma3/2  # in units of hbar/a
    v4 = jnp.sqrt(3)*gamma4/2  # in units of hbar/a

    # Assuming K valley (xi=1) for the base function. 
    # If xi=-1, caller should pass -kx.
    pi = kx + 1j * ky

    t      = (-v**2/gamma1 - v3/(4*jnp.sqrt(3))) * pi**2 + v3 * jnp.conj(pi)
    u_symm = (2*v*v4/gamma1 + delta_prime * v**2/gamma1**2) * jnp.abs(pi)**2
    u_asym = -U/2 * (1- 2*v**2/gamma1**2 * jnp.abs(pi)**2) + delta_AB/2

    M = jnp.array([
        [u_symm + u_asym,  jnp.conj(t)],
        [t,                u_symm - u_asym]
    ])

    return M

def get_hamiltonian_2bands(params: dict = graphene_params_BLG):
    h_func = partial(hamiltonian_2bands, params=params)
    return h_func
