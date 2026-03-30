# contimod_graphene

`contimod_graphene` is a standalone Python package for multilayer graphene Hamiltonians and related single-particle utilities.

It is designed to be useful directly in its own right, with `contimod` treated as one downstream consumer rather than a required companion package.

Current scope includes:
- **Bernal (ABA) stacking**
- **Rhombohedral (ABC) stacking**
- Immutable, JSON-backed parameter sets
- Standalone model objects with thin wrappers over the kernel layer
- Basis and symmetry helpers
- JAX-friendly batched Hamiltonian evaluation

It includes functionality for both zero-field Hamiltonians and Landau Level (LL) Hamiltonians.

The maintained standalone starting points are:
- the top-level public API shown below
- [docs/usage.md](/Users/wolft/Dev/contimod_graphene/docs/usage.md)
- [examples/standalone_gallery.py](/Users/wolft/Dev/contimod_graphene/examples/standalone_gallery.py)
- [examples/standalone_quickstart.py](/Users/wolft/Dev/contimod_graphene/examples/standalone_quickstart.py)

## Relationship To `contimod`

`contimod_graphene` owns the low-level graphene-model layer: Hamiltonian builders, parameter presets, basis metadata, and related lightweight helpers.

`contimod` builds on top of that layer for discretization, mesh/state containers, and many-body workflows such as SCF, susceptibility, TDHF, and superconductivity. Using `contimod` is optional, not required.

## Quick Start

The canonical user-facing surface is the top-level parameter/model API:

```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

H = model.hamiltonian(0.1, 0.0)
Hs = model.hamiltonian_batch([[0.0, 0.0], [0.1, 0.0]])
Hll = model.landau_level_hamiltonian(10.0, n_cut=40, valley="K")
H2 = model.two_band_hamiltonian(0.1, 0.0)
```

Runnable standalone examples live in `examples/standalone_gallery.py`, and the docs mirror that surface in `docs/usage.md` and `docs/examples.md`.

The main public entry points are:
- `GrapheneTBParameters`
- `load_parameter_set(name_or_path)`
- `list_parameter_sets()`
- `BernalMultilayer`
- `RhombohedralMultilayer`

Physicist-friendly aliases are also available:
- `ABAMultilayer`
- `ABCMultilayer`

For a slightly longer walkthrough, see [docs/usage.md](/Users/wolft/Dev/contimod_graphene/docs/usage.md). For maintained example material, start with [examples/standalone_quickstart.py](/Users/wolft/Dev/contimod_graphene/examples/standalone_quickstart.py) and [examples/README.md](/Users/wolft/Dev/contimod_graphene/examples/README.md).

## Low-Level Modules

The low-level kernel modules remain available for advanced use, JAX-focused workflows, and direct access to the functional core:
- `contimod_graphene.bernal`
- `contimod_graphene.rhombohedral`
- `contimod_graphene.params`
- `contimod_graphene.basis`
- `contimod_graphene.symmetry`
- `contimod_graphene.utils`

## Installation

### Option 1 (developer install)

Clone the repository and install it in editable mode:
```bash
pip install -e ".[dev]"
```
This gives you the package locally as `import contimod_graphene as cg`.

If you prefer `hatch`, create and enter the managed development environment with:
```bash
hatch env create
hatch shell
```

Quick import check:
```bash
python -c "import contimod_graphene as cg; print(cg.list_parameter_sets())"
```

### Option 2 (GitHub install)

If you only want to use the package, install it directly from GitHub:
```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```
This installs the same top-level import: `import contimod_graphene as cg`.

### Option 3 (cross-platform)

*Note: This method is not tested on all platforms. It may or may not fail on arm-based systems (such as Apple Silicon).*

Third-party requirements:
 - Docker
 - repo2docker

To spin up a containerized jupyter environment, run:
```bash
$ jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```

While this is the fastest way to spin up a container, you can also containerize this package yourself.

## Credit 
This package is developed and maintained by Dr. Tobias Wolf. Feel free to contact us, and please give us credit if you use this work. 
