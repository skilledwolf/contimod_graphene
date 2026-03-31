# contimod_graphene

`contimod_graphene` is a standalone Python package for multilayer graphene tight-binding Hamiltonians, parameter sets, basis metadata, and related single-particle utilities.

Current scope includes:
- **Bernal (ABA) stacking**
- **Rhombohedral (ABC) stacking**
- Immutable, JSON-backed parameter sets
- Standalone model objects with thin wrappers over the kernel layer
- Basis and symmetry helpers
- JAX-friendly batched Hamiltonian evaluation

It includes both zero-field Hamiltonians and Landau-level (LL) Hamiltonians.

The maintained starting points are:
- the top-level public API shown below
- [docs/usage.md](https://github.com/skilledwolf/contimod_graphene/blob/main/docs/usage.md)
- [examples/standalone_gallery.py](https://github.com/skilledwolf/contimod_graphene/blob/main/examples/standalone_gallery.py)
- [examples/standalone_quickstart.py](https://github.com/skilledwolf/contimod_graphene/blob/main/examples/standalone_quickstart.py)

For examples/tests on this machine, prefer `JAX_PLATFORMS=cpu`; the Apple Metal backend still hits known JAX failures in this repo.

## Quick Start

For an ABC-trilayer Hamiltonian, its low-energy 2-band reduction, and a bilayer LL matrix:

```python
import numpy as np
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0, Delta=0.0)
abc = cg.RhombohedralMultilayer(n_layers=3, params=params)
ab = cg.BernalMultilayer(n_layers=2)

print(abc.hamiltonian(0.1, 0.0).shape)
print(np.round(np.linalg.eigvalsh(np.asarray(abc.two_band_hamiltonian(0.02, 0.0))), 3))
print(ab.landau_level_hamiltonian(10.0, n_cut=6, valley="K").shape)
```

```text
(6, 6)
[-10.178  11.608]
(22, 22)
```

If you want the maintained script rather than a pasted snippet:

```bash
JAX_PLATFORMS=cpu python examples/standalone_quickstart.py
```

The built-in ABC/TLG preset carries `U=30.0` meV and `Delta=-1.15` meV. The quickstart pins `Delta=0.0` so the example isolates the outer-layer bias `U`.

The main public entry points are:
- `GrapheneTBParameters`
- `load_parameter_set(name_or_path)`
- `list_parameter_sets()`
- `BernalMultilayer`
- `RhombohedralMultilayer`

Physicist-friendly aliases are also available:
- `ABAMultilayer`
- `ABCMultilayer`

For a slightly longer walkthrough with equations, conventions, and more outputs, see [docs/usage.md](https://github.com/skilledwolf/contimod_graphene/blob/main/docs/usage.md). For maintained example material, start with [examples/standalone_quickstart.py](https://github.com/skilledwolf/contimod_graphene/blob/main/examples/standalone_quickstart.py), [docs/examples.md](https://github.com/skilledwolf/contimod_graphene/blob/main/docs/examples.md), and [examples/README.md](https://github.com/skilledwolf/contimod_graphene/blob/main/examples/README.md).

## Physics At A Glance

The package exposes three common surfaces:

$$
H(\mathbf{k}) \psi_{n\mathbf{k}} = E_n(\mathbf{k}) \psi_{n\mathbf{k}}
$$

$$
H^{ABC_N}_{2\mathrm{band}} \propto
\begin{pmatrix}
0 & (\pi^\dagger)^N \\
\pi^N & 0
\end{pmatrix},
\qquad E \propto k^N
$$

$$
\dim H_{\mathrm{LL}} = n_{\mathrm{layers}} \left(2 n_{\mathrm{cut}} - 1\right)
$$

Useful parameter conventions:
- Bernal `Delta` is the package A/B sublattice mass term, while Bernal `delta` is the dimer/non-dimer onsite offset.
- Rhombohedral `Delta` matches the usual trilayer `Δ2` meaning for `n_layers=3`; for thicker stacks it is reused as a package-specific inversion-even layer-curvature parameter.
- LL builders return dense matrices, so the size formula above matters quickly when you increase `n_cut`.

## Low-Level Modules

The low-level kernel modules remain available for advanced use, JAX-focused workflows, and direct access to the functional core:
- `contimod_graphene.bernal`
- `contimod_graphene.rhombohedral`
- `contimod_graphene.params`
- `contimod_graphene.basis`
- `contimod_graphene.symmetry`
- `contimod_graphene.utils`

## Installation

### Use the package

Install directly from GitHub:

```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```

Quick smoke check:

```bash
python -c "import contimod_graphene as cg; print(cg.list_parameter_sets())"
```

### Develop locally

Clone the repository and install it in editable mode:
```bash
pip install -e ".[dev]"
```

If you prefer `hatch`, create and enter the managed development environment with:
```bash
hatch env create
hatch shell
```

### Containerized Jupyter

If you want a throwaway Jupyter environment, use `repo2docker`:

```bash
jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```

Requirements: Docker plus `repo2docker`. This path is convenient, but not tested on every platform.

## Credit 
This package is developed and maintained by Dr. Tobias Wolf. Feel free to contact us, and please give us credit if you use this work. 

## Downstream Note

`contimod_graphene` is intended to stand on its own. A separate downstream package, `contimod`, builds on top of this model layer for discretization, mesh/state containers, and many-body workflows, but it is not required for normal use here.
