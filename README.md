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

The main public entry points are:
- `GrapheneTBParameters`
- `load_parameter_set(name_or_path)`
- `list_parameter_sets()`
- `BernalMultilayer`
- `RhombohedralMultilayer`

Physicist-friendly aliases are also available:
- `ABAMultilayer`
- `ABCMultilayer`

## Low-Level Modules

The low-level kernel modules remain available for advanced use, JAX-focused workflows, and direct access to the functional core:
- `contimod_graphene.bernal`
- `contimod_graphene.rhombohedral`
- `contimod_graphene.params`
- `contimod_graphene.basis`
- `contimod_graphene.symmetry`
- `contimod_graphene.utils`

## Installation

### Option 1a (for developers)

You can `git clone` the repository, activate your python environment of choice and install the package as editable using
```bash
pip install -e .
```
This will allow you to use `import contimod_graphene as cm_graphene` in your python code. 
You can uninstall the package any with `pip uninstall contimod_graphene`.

If you plan to contribute to the package, you must learn how to use `git` and how to create pull requests.

### Option 1b (for developers)

We now use [`hatch`](https://github.com/pypa/hatch) as dev tool, which you need to install seperately. It automates the entire development process. A suitable environment can for example be created using
```bash
hatch env create
hatch shell
```

### Option 2 (preferred method for users)

If you are sure that you will not need to modify the package, then open the terminal and run
```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```
This will allow you to do `import contimod_graphene as cm_graphene` in your python code. You can uninstall the package with `pip uninstall contimod_graphene`.

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
