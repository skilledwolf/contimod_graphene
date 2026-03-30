# Public API Plan

This document captures the recommended first implementation pass for turning `contimod_graphene` into a clearer standalone package with a more coherent public API and parameter-management layer.

## Goals
- Keep the numerically important Hamiltonian kernels pure and easy to JIT / `vmap`.
- Present a more user-friendly public surface than "import module, pass raw dict, remember layer-family conventions manually".
- Make model identity explicit: stacking family, layer count, and parameter set should travel together.
- Improve parameter management without giving up JSON-backed presets.
- Preserve good interoperability with JAX and future JAX-based NN libraries.

## Non-Goals
- Do not move many-body workflows here from `contimod`.
- Do not hide the low-level kernel modules entirely; advanced users may still want them.
- Do not add large optional dependencies unless they unlock a clearly intended package surface.

## Recommended Public API

### Canonical user-facing names
- `GrapheneTBParameters`
- `load_parameter_set(name_or_path)`
- `list_parameter_sets()`
- `BernalMultilayer`
- `RhombohedralMultilayer`

Physicist-friendly aliases are acceptable if they stay secondary and do not multiply the surface too much:
- `ABAMultilayer` as an alias of `BernalMultilayer`
- `ABCMultilayer` as an alias of `RhombohedralMultilayer`

The docs should pick one canonical style and stick to it. Recommended canonical style: `BernalMultilayer` / `RhombohedralMultilayer`, with ABA/ABC terminology explained alongside.

### Example target usage
```python
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

H = model.hamiltonian(0.1, 0.0)
Hs = model.hamiltonian_batch(ks)
Hll = model.landau_level_hamiltonian(B=10.0, n_cut=40, valley="K")
H2 = model.two_band_hamiltonian(0.1, 0.0)
```

## Public API Shape

### `GrapheneTBParameters`
Recommended contract:
- immutable
- easy to inspect (`repr`, `.to_dict()`)
- easy to modify (`.replace(**kwargs)`)
- JSON round-trip friendly (`.from_json(path)`, `.to_json(path)`)
- preset-aware (`.preset("blg")`)
- family-validated (`validate_for("bernal")`, `validate_for("rhombohedral")`, or family-specific validation inside model constructors)

Notes:
- A frozen dataclass or a small validated mapping type are both acceptable.
- Prefer explicit fields over a bare untyped dict for the public surface.
- If extensibility is needed, keep an `extras: dict[str, float]` field rather than silently accepting arbitrary top-level keys.

### `BernalMultilayer` / `RhombohedralMultilayer`
Recommended contract:
- immutable lightweight model object
- stores `n_layers` and `params`
- methods are thin wrappers over pure functional kernels
- no file I/O, alias resolution, or doc-driven branching inside traced code

Recommended methods:
- `.hamiltonian(kx, ky)`
- `.hamiltonian_batch(ks)`
- `.landau_level_hamiltonian(B, *, n_cut, valley="K")`
- `.two_band_hamiltonian(kx, ky)` where meaningful
- `.with_params(**overrides)` or `.replace(params=...)`

Optional metadata:
- `.family`
- `.stacking_label`
- `.default_preset_name`

## Functional Core Requirements

The OO layer must stay thin. The actual computational core should remain pure functions roughly like:

```python
hamiltonian(kx, ky, *, n_layers, params)
hamiltonian_2bands(kx, ky, *, n_layers, params)
hamiltonian_LL(B, *, n_layers, n_cut, flip_valley, params)
```

This is important for:
- `jax.jit`
- `jax.vmap`
- differentiation with respect to physical parameters
- later use with JAX NN libraries

## JAX / NN Compatibility Guidelines

To keep the package compatible with JAX-first workflows:
- keep kernel functions pure
- keep model/parameter objects immutable
- make parameter objects PyTree-friendly if practical
- avoid hidden Python-side state changes in model methods
- keep validation and preset loading outside traced paths
- ensure numeric parameter values can be provided as Python floats or JAX arrays

If trainable parameter workflows become important later, `GrapheneTBParameters` should either:
- be registered as a PyTree, or
- provide a cheap `.to_dict()` / `.from_dict()` path compatible with JAX transforms

## Recommended File Layout

First implementation pass:
- leave `bernal.py` and `rhombohedral.py` in place as the low-level kernel modules
- add `models.py` for the user-facing model objects
- evolve `params.py` into the stronger public parameter layer
- keep `basis.py` and `symmetry.py`
- move `batch_hamiltonian` out of `utils.py` into a clearer home if needed, or keep it but document that ownership explicitly

Conservative first-cut layout:
```text
src/contimod_graphene/
  __init__.py
  params.py
  models.py
  bernal.py
  rhombohedral.py
  basis.py
  symmetry.py
  utils.py
```

Possible later cleanup after the surface settles:
```text
src/contimod_graphene/
  __init__.py
  api.py
  parameters.py
  models.py
  kernels/
    __init__.py
    bernal.py
    rhombohedral.py
  basis.py
  symmetry.py
  numerics.py
```

## Parameter-Management Recommendations

### Current issues to address
- raw dicts are the de facto public contract
- family/default semantics are too implicit
- rhombohedral builders currently default to BLG presets, which weakens model identity
- users have to infer which keys matter from kernel internals

### Recommended first-cut parameter API
```python
params = GrapheneTBParameters.preset("tlg")
params = params.replace(U=20.0, gamma3=310.0)
params.to_json("my_params.json")
params2 = GrapheneTBParameters.from_json("my_params.json")
```

Recommended helpers:
- `GrapheneTBParameters.preset(name)`
- `load_parameter_set(name_or_path)`
- `list_parameter_sets()`

Recommended validation behavior:
- unknown preset names raise a clear error
- overlaying unknown keys is explicit, not silent
- model constructors validate that the parameter object is compatible with the chosen family

## Recommended Implementation Order

1. Fix the existing default/doc mismatches in `rhombohedral.py`.
2. Consolidate duplicated helper ownership between `utils.py` and `basis.py`.
3. Implement `GrapheneTBParameters` in `params.py` while preserving existing preset-loading behavior.
4. Implement `BernalMultilayer` / `RhombohedralMultilayer` in `models.py` as thin wrappers over the existing kernels.
5. Export the new surface from `__init__.py` and update README/examples to use it.
6. Add focused tests for the new public API and parameter semantics.

## Validation Expectations
- import-level smoke tests for the new public API
- parameter preset / alias / overlay / round-trip tests
- model constructor tests
- regression checks that new model methods call through to the existing kernels correctly
- keep current kernel-shape/hermiticity tests, but add more behavior-level checks where practical

## Optional Dependencies
- `ase`: only if the package intends to own BZ/path helpers or simple plotting-ready band-path utilities
- `spglib`: only if symmetry classification becomes part of the intended public API

Neither should block the first pass of the public API / parameter cleanup.
