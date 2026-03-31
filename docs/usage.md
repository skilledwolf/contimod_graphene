# Usage Guide

`contimod_graphene` is organized around two public concepts:
- `GrapheneTBParameters` for validated tight-binding parameters
- `BernalMultilayer` / `RhombohedralMultilayer` for thin, immutable model objects

The low-level `bernal` and `rhombohedral` modules remain available, but the default usage story should start from the top-level API.

For runnable standalone scripts that mirror the examples below, see `examples/standalone_gallery.py`.

Maintained standalone scripts:
- `examples/standalone_quickstart.py`
- `examples/standalone_gallery.py`

## Basic Usage

### 1. Bernal (ABA) model

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("blg").replace(U=0.0)
model = cg.BernalMultilayer(n_layers=2, params=params)

H = model.hamiltonian(0.05, 0.0)
print(H.shape)  # (4, 4)
```

### 2. Rhombohedral (ABC) model

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

H = model.hamiltonian(0.1, 0.1)
print(H.shape)  # (6, 6)
```

The built-in ABC/TLG preset itself carries finite asymmetry terms (`U=30.0` meV and
`Delta=-1.15` meV), so standalone examples that want a symmetry-restored starting
point should pin those explicitly as above.

If you omit `params`, the model objects use family-appropriate built-in defaults:
- `BernalMultilayer()` defaults to the BLG preset
- `RhombohedralMultilayer()` defaults to the ABC/TLG preset, including its built-in
  `U=30.0` meV and `Delta=-1.15` meV offsets

### 3. Landau levels

```python
import contimod_graphene as cg

model = cg.BernalMultilayer(n_layers=2)
H_LL = model.landau_level_hamiltonian(10.0, n_cut=20, valley="K")
print(H_LL.shape)
```

### 4. Batched evaluation

```python
import jax.numpy as jnp
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

Hs = model.hamiltonian_batch(ks)
bands = jnp.linalg.eigvalsh(Hs)
```

### 5. Two-band models

```python
import contimod_graphene as cg

abc_params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
abc = cg.RhombohedralMultilayer(n_layers=3, params=abc_params)
H2_abc = abc.two_band_hamiltonian(0.02, -0.01)

ab = cg.BernalMultilayer(n_layers=2)
H2_ab = ab.two_band_hamiltonian(0.02, -0.01)
```

The Bernal two-band reduction is only implemented for bilayer graphene.

## Parameters

Built-in presets are available through either the classmethod or the module-level constants:

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("4lg")
same = cg.graphene_params_4LG
```

Useful helpers:
- `cg.load_parameter_set("tlg")`
- `cg.load_parameter_set("path/to/custom.json")`
- `cg.list_parameter_sets()`

Parameter objects are immutable mappings, so they work with both the new model surface and the low-level kernels:

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=15.0, Delta=0.0, lambda1_eff=1.0)

print(params["U"])
print(dict(params))
params.to_json("my_params.json")
same_params = cg.GrapheneTBParameters.from_json("my_params.json")
```

Explicit overrides of unknown keys are stored as `extras`, which is useful for downstream workflows that carry additional metadata or symmetry-breaking couplings alongside the core tight-binding parameters.

## Low-Level Functional API

Advanced users can still call the low-level modules directly:

```python
from contimod_graphene import bernal, rhombohedral
from contimod_graphene import graphene_params_BLG, graphene_params_TLG

H_ab = bernal.hamiltonian(0.05, 0.0, n_layers=2, params=graphene_params_BLG)
H_abc = rhombohedral.hamiltonian(
    0.05,
    0.0,
    n_layers=3,
    params=graphene_params_TLG.replace(U=0.0, Delta=0.0),
)
```

These functions remain the computational core that the model objects wrap.

## Units and Conventions

### Parameter naming

- **Bernal `delta`** is the dimer/non-dimer onsite offset. This is the package knob that is closest to the bilayer literature's `Δ'` convention.
- **Bernal `Delta`** is a package-defined A/B sublattice mass term `(+Delta/2 on A, -Delta/2 on B)`. For exact even-layer Bernal inversion-symmetry tests, pin `U=0.0` and `Delta=0.0`.
- **Rhombohedral `Delta`** matches the standard trilayer `Δ2` meaning only for `n_layers=3`, where it shifts the middle layer relative to the average outer-layer energy. For `n_layers>3`, the package reuses the same scalar as a mean-zero inversion-even layer-curvature parameter. That thicker-stack meaning is a package convention, not a standard literature parameterization.
- **Rhombohedral `delta`** is currently accepted for shared-parameter compatibility but is intentionally unused by the rhombohedral kernels.

- **Energies**: tight-binding parameters are in meV.
- **Momentum**: `kx` and `ky` are in inverse-length units consistent with the chosen parameterization.
- **Magnetic field**: `B` is in Tesla.
- **LL matrix size**: the Landau-level helpers return dense matrices of shape `(n_layers * (2*n_cut - 1), n_layers * (2*n_cut - 1))`.
- **JAX vs NumPy**:
  - zero-field Hamiltonians are JAX arrays and work with `jax.jit` / `jax.vmap`
  - Landau-level Hamiltonians are dense host-side arrays by default
- **Local validation on this machine**: use CPU for tests (`JAX_PLATFORMS=cpu`) because the Apple Metal backend currently hits known unsupported JAX paths in this repo.
