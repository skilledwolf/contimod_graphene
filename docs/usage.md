# Usage Guide

`contimod_graphene` provides tools to construct Hamiltonians for multilayer graphene systems. It supports both **Bernal (ABA)** and **Rhombohedral (ABC)** stacking configurations.

## Basic Usage

### 1. Bernal (ABA) Stacking

To simulate Bernal-stacked graphene (e.g., Bilayer Graphene), use the `bernal` module.

```python
import numpy as np
import contimod_graphene.bernal as bernal
from contimod_graphene.params import graphene_params_BLG

# 1. Define system parameters
n_layers = 2
params = graphene_params_BLG

# 2. Get the Hamiltonian function
# This returns a JIT-compiled function h(kx, ky)
h_func = bernal.get_hamiltonian(n_layers=n_layers, params=params)

# 3. Evaluate the Hamiltonian at a specific momentum point
kx, ky = 0.05, 0.0  # Momentum in 1/Angstrom (approx, depends on units of parameters)
H = h_func(kx, ky)

print(f"Hamiltonian shape: {H.shape}")
# Output: (4, 4) for 2 layers * 2 sublattices
```

### 2. Rhombohedral (ABC) Stacking

For Rhombohedral stacking, use the `rhombohedral` module.

```python
import contimod_graphene.rhombohedral as rhombohedral
from contimod_graphene.params import graphene_params_TLG

# 1. Define system parameters (e.g., Trilayer)
n_layers = 3
params = graphene_params_TLG

# If you omit `params`, the rhombohedral convenience wrappers default to
# the ABC trilayer preset.

# 2. Get the Hamiltonian function
h_func = rhombohedral.get_hamiltonian(n_layers=n_layers, params=params)

# 3. Evaluate
H = h_func(0.1, 0.1)
print(f"Hamiltonian shape: {H.shape}")
# Output: (6, 6)
```

### 3. Landau Levels

You can also construct Landau Level (LL) Hamiltonians in a magnetic field.

```python
import contimod_graphene.bernal as bernal
from contimod_graphene.params import graphene_params_BLG

# Calculate LL Hamiltonian for Bernal bilayer in a 10 Tesla field
B_field = 10.0 # Tesla
N_cut = 20     # Number of Landau levels to include

h_ll_func = bernal.get_hamiltonian_LL(
    n_layers=2, 
    n_cut=N_cut, 
    flip_valley=False, 
    params=graphene_params_BLG
)

H_LL = h_ll_func(B_field)
print(f"LL Hamiltonian shape: {H_LL.shape}")
```

### 4. Vectorized evaluation with `jax.vmap`

Because the zero-field Hamiltonians are JAX functions, you can efficiently evaluate them on a whole path of momenta using `jax.vmap`. For example, to compute a simple band structure along a 1D path:

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import contimod_graphene.rhombohedral as rhombohedral
from contimod_graphene.params import graphene_params_TLG

# Trilayer ABC parameters and Hamiltonian
n_layers = 3
params = graphene_params_TLG
h_func = rhombohedral.get_hamiltonian(n_layers=n_layers, params=params)

# 1D k-path along kx
k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

# Evaluate H(k) and diagonalize for each k
Hs = jax.vmap(h_func, in_axes=(0, 0))(*ks.T)
bands = jnp.linalg.eigvalsh(Hs)

for band in bands.T:
    plt.plot(k_lin, band, color="black", linewidth=1.0, alpha=0.7)

plt.xlabel(r"$k_x\,a$")
plt.ylabel("Energy [meV]")
plt.show()
```

## Parameters

The package includes standard parameter sets in `contimod_graphene.params`:

*   `graphene_params`: Single layer
*   `graphene_params_BLG`: Bilayer (Bernal)
*   `graphene_params_TLG`: Trilayer (Rhombohedral/Bernal)
*   `graphene_params_4LG`: 4-Layer

You can also provide your own dictionary of parameters:

```python
custom_params = {
    "gamma0": 3100,
    "gamma1": 380,
    "gamma2": -15,
    "gamma3": 300,
    "gamma4": 140,
    "U": 0.0,
    "delta": 10.0,
    "Delta": 0.0
}

h_func = bernal.get_hamiltonian(n_layers=2, params=custom_params)
```

## Units and Conventions

- **Energies**: All tight-binding parameters in `contimod_graphene.params` are given in milli-electron volts (meV).
- **Momentum**: The arguments `kx` and `ky` are measured in inverse length units consistent with your parameterization (typically Å⁻¹ for the provided parameter sets).
- **Magnetic field**: The variable `B` in Landau-level Hamiltonians is in Tesla.
- **LL matrix size**: For both Bernal and Rhombohedral stackings, `get_hamiltonian_LL(n_layers, n_cut, ...)` returns a dense matrix of shape `(n_layers * (2*n_cut - 1), n_layers * (2*n_cut - 1))`, reflecting the asymmetric basis with `(N_A, N_B) = (n_cut-1, n_cut)` in one valley and swapped in the other.
- **JAX vs NumPy**:
  - Zero-field Hamiltonians returned by `get_hamiltonian` and `get_2band_hamiltonian` are JAX functions and can be used with `jax.jit` and `jax.vmap`.
  - Landau-level Hamiltonians returned by `get_hamiltonian_LL` are NumPy arrays constructed on the host and are not JAX-traceable by default.
