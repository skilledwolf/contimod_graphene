# contimod_graphene: Helper package for contimod

`contimod_graphene` provides Hamiltonian construction tools for multilayer graphene systems, specifically supporting:
- **Bernal (ABA) stacking**
- **Rhombohedral (ABC) stacking**

It includes functionality for both zero-field Hamiltonians and Landau Level (LL) Hamiltonians.

## Modules

### `contimod_graphene.bernal`
Provides Hamiltonians for Bernal-stacked (ABA) multilayer graphene.
- `get_hamiltonian(n_layers, params)`: Returns a function for the zero-field Hamiltonian.
- `get_hamiltonian_LL(n_layers, n_cut, flip_valley, params)`: Returns a function for the Landau Level Hamiltonian.

### `contimod_graphene.rhombohedral`
Provides Hamiltonians for Rhombohedral-stacked (ABC) multilayer graphene.
- `get_hamiltonian(n_layers, params)`: Returns a function for the zero-field Hamiltonian.
- `get_2band_hamiltonian(n_layers, params)`: Returns a function for the effective 2-band Hamiltonian.
- `get_hamiltonian_LL(n_layers, n_cut, flip_valley, params)`: Returns a function for the Landau Level Hamiltonian.

### `contimod_graphene.params`
JSON-backed parameter sets for graphene tight-binding models (see `data/params.json`).
- `get_params(kind_or_dict)`: Return params by preset name/alias (e.g., `"slg"`, `"blg"`, `"tlg"`, `"4lg"`), a JSON file path, or pass through a dict.
- `load(path)`: Load params from a JSON file.
- `list_sets()`: List available presets.

### `contimod_graphene.utils`
Helper functions for Hamiltonian construction.
- `extract_params(params, keys)`: Extract specific parameters from a dictionary.
- `layer_coordinates(n_layers)`: Get z-coordinates for layers.
- `sublattice_coordinates(n_layers)`: Get sublattice indices.
- `construct_ll_ops(N_A, N_B)`: Construct ladder operators for Landau Level calculations.
- `batch_hamiltonian(h_fn, jit=True)`: Vectorize a single-k Hamiltonian over an array of k-points.

### `contimod_graphene.basis`
Basis helpers (layer/sublattice coordinates and operator construction) used by downstream wrappers.

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
