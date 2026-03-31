# Usage Guide

`contimod_graphene` revolves around two public concepts:
- `GrapheneTBParameters` for validated tight-binding parameters
- `BernalMultilayer` / `RhombohedralMultilayer` for thin, immutable model objects

If you want one maintained smoke test instead of reading first:

```bash
JAX_PLATFORMS=cpu python examples/standalone_quickstart.py
```

That writes `examples/outputs/standalone_quickstart_summary.json`. On this machine, CPU is the reliable backend for examples and tests; Apple Metal still hits known JAX backend failures in this repo.

## What The Package Computes

At zero field, the main object is the multilayer tight-binding Hamiltonian

$$
H(\mathbf{k}) \psi_{n\mathbf{k}} = E_n(\mathbf{k}) \psi_{n\mathbf{k}},
\qquad \mathbf{k} = (k_x, k_y).
$$

In a perpendicular magnetic field, the Landau-level helpers return the dense LL-basis matrix

$$
H_{\mathrm{LL}}(B) \phi_{n,B} = E_n(B) \phi_{n,B}.
$$

For rhombohedral stacks, `two_band_hamiltonian(...)` exposes the low-energy two-band reduction. Schematically, near neutrality,

$$
H^{ABC_N}_{2\mathrm{band}}(\pi) \sim
\begin{pmatrix}
U/2 & (\pi^\dagger)^N \\
\pi^N & -U/2
\end{pmatrix},
\qquad \pi = \xi k_x + i k_y,
$$

up to the usual velocity/hopping prefactors and remote-hopping corrections. The clean low-energy scaling is therefore $E \propto k^N$.

Useful shape rules:

| Call | Physics object | Returned shape |
| --- | --- | --- |
| `model.hamiltonian(kx, ky)` | zero-field tight-binding matrix | `(2 * n_layers, 2 * n_layers)` |
| `model.hamiltonian_batch(ks)` | stack of zero-field matrices | `(n_k, 2 * n_layers, 2 * n_layers)` |
| `model.two_band_hamiltonian(kx, ky)` | low-energy effective model | `(2, 2)` |
| `model.landau_level_hamiltonian(B, n_cut=...)` | dense LL-basis matrix | `(n_layers * (2 * n_cut - 1), n_layers * (2 * n_cut - 1))` |

## 30-Second API Example

```python
import json
import numpy as np
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0, Delta=0.0)
abc = cg.RhombohedralMultilayer(n_layers=3, params=params)
ab = cg.BernalMultilayer(n_layers=2)

summary = {
    "zero_field_shape": list(np.asarray(abc.hamiltonian(0.1, 0.0)).shape),
    "two_band_eigs_meV": [
        round(float(x), 3)
        for x in np.linalg.eigvalsh(np.asarray(abc.two_band_hamiltonian(0.02, 0.0)))
    ],
    "landau_level_shape": list(
        np.asarray(ab.landau_level_hamiltonian(10.0, n_cut=6, valley="K")).shape
    ),
}
print(json.dumps(summary, indent=2))
```

```json
{
  "zero_field_shape": [6, 6],
  "two_band_eigs_meV": [-10.178, 11.608],
  "landau_level_shape": [22, 22]
}
```

The built-in ABC/TLG preset carries `U=30.0` meV and `Delta=-1.15` meV. Pin `Delta=0.0` when you want the example to isolate the outer-layer bias `U`.

`examples/standalone_quickstart.py` writes a larger JSON summary with band extrema, sample full-model eigenvalues, and LL sample energies if you want a file-backed reference output rather than an inline snippet.

## Common Tasks

### Bernal / ABA Zero-Field Hamiltonian

```python
import numpy as np
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("blg").replace(U=0.0)
model = cg.BernalMultilayer(n_layers=2, params=params)

H = np.asarray(model.hamiltonian(0.05, 0.0))
print(H.shape)
```

```text
(4, 4)
```

### Rhombohedral / ABC Zero-Field Hamiltonian

```python
import numpy as np
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

evals = np.linalg.eigvalsh(np.asarray(model.hamiltonian(0.1, 0.1)))
print(np.round(evals, 3))
```

```text
[-630.023 -363.613   -5.393    5.393  363.613  630.023]
```

If you omit `params`, the default presets are family-specific:
- `BernalMultilayer()` defaults to the BLG preset
- `RhombohedralMultilayer()` defaults to the ABC/TLG preset

### Batched Band Scan

```python
import numpy as np
import jax.numpy as jnp
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 5)
ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

Hs = np.asarray(model.hamiltonian_batch(ks, jit=False))
bands = np.linalg.eigvalsh(Hs)
print(Hs.shape)
print(bands.shape)
```

```text
(5, 6, 6)
(5, 6)
```

### Landau Levels

```python
import numpy as np
import contimod_graphene as cg

model = cg.BernalMultilayer(n_layers=2)
H_LL = np.asarray(model.landau_level_hamiltonian(10.0, n_cut=20, valley="K"))
print(H_LL.shape)
```

```text
(78, 78)
```

For `n_layers=2`, the Bernal and rhombohedral kernels describe the same AB bilayer connectivity, but the maintained public entry point in this package is `BernalMultilayer`.

### Two-Band Models

```python
import numpy as np
import contimod_graphene as cg

abc_params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0, Delta=0.0)
abc = cg.RhombohedralMultilayer(n_layers=3, params=abc_params)
ab = cg.BernalMultilayer(n_layers=2)

print(np.asarray(abc.two_band_hamiltonian(0.02, -0.01)).shape)
print(np.asarray(ab.two_band_hamiltonian(0.02, -0.01)).shape)
```

```text
(2, 2)
(2, 2)
```

The Bernal two-band reduction is only implemented for bilayer graphene.

## Parameters

Built-in presets are available through either the classmethod or the module-level constants:

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("4lg")
same = cg.graphene_params_4LG

print(cg.list_parameter_sets())
print(params["gamma1"])
```

```text
['slg', 'blg', 'tlg', '4lg']
370.0
```

Typical parameter workflow:

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(
    U=15.0,
    Delta=0.0,
    lambda1_eff=1.0,
)

params.to_json("my_params.json")
same_params = cg.GrapheneTBParameters.from_json("my_params.json")
print(params["U"], same_params["U"])
```

```text
15.0 15.0
```

Useful helpers:
- `cg.load_parameter_set("tlg")`
- `cg.load_parameter_set("path/to/custom.json")`
- `cg.list_parameter_sets()`

Unknown override keys are stored in `extras`, which is useful if you want to carry downstream metadata or symmetry-breaking couplings alongside the core tight-binding parameters.

## Low-Level Functional API

Advanced users can still call the low-level kernels directly:

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

These functions are the computational core wrapped by the model objects.

## Conventions That Matter

Parameter naming:

| Surface | Meaning |
| --- | --- |
| Bernal `delta` | dimer/non-dimer onsite offset; closest to the bilayer-literature `Δ'` convention |
| Bernal `Delta` | package A/B sublattice mass term `(+Delta/2 on A, -Delta/2 on B)` |
| Rhombohedral `Delta` | standard trilayer `Δ2` meaning for `n_layers=3`; package-specific inversion-even layer-curvature parameter for `n_layers>3` |
| Rhombohedral `delta` | accepted for shared-parameter compatibility but intentionally unused by the rhombohedral kernels |

General units and backend notes:

| Quantity | Convention |
| --- | --- |
| Energies | meV |
| Momenta `kx`, `ky` | inverse-length units consistent with the chosen parameterization |
| Magnetic field `B` | Tesla |
| Zero-field Hamiltonians | JAX arrays that work with `jax.jit` / `jax.vmap` |
| LL Hamiltonians | dense host-side arrays by default |
