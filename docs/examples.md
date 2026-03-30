# Standalone Examples

The repository still carries a few historical notebooks in `examples/`, but the maintained standalone entry points now live in `examples/standalone_quickstart.py` and `examples/standalone_gallery.py`. The first three workflows below are direct excerpts from the gallery script and require `contimod_graphene` plus `matplotlib`.

## 1. Rhombohedral (ABC) Band Structures (`examples/standalone_gallery.py`)

This example compares the full tight-binding model with its low-energy 2-band approximation for ABC trilayer graphene.

The figure below shows an example band structure for ABC trilayer graphene with an
interlayer bias. The maintained example pins `Delta=0.0` so the plot isolates the
effect of `U` rather than the ABC/TLG preset's small built-in `Delta=-1.15` offset:

![ABC trilayer band structure](_static/abc_trilayer_bandstructure.png)

**Example: full vs 2-band model for ABC trilayer**

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("tlg").replace(U=10.0, Delta=0.0)
model = cg.RhombohedralMultilayer(n_layers=3, params=params)

k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

Hs_full = model.hamiltonian_batch(ks)
bands_full = jnp.linalg.eigvalsh(Hs_full)

Hs_low = jax.vmap(
    lambda kx, ky: model.two_band_hamiltonian(kx, ky),
    in_axes=(0, 0),
)(*ks.T)
bands_low = jnp.linalg.eigvalsh(Hs_low)

fig, ax = plt.subplots()
for band in bands_full.T:
    ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)
for band in bands_low.T:
    ax.plot(k_lin, band, color="red", linewidth=1.5, linestyle="--", alpha=0.9)

ax.set_xlabel(r"$k_x\,a$")
ax.set_ylabel("Energy [meV]")
ax.set_ylim(-80, 80)
plt.show()
```

## 2. Landau Level Fan Diagrams (`examples/standalone_gallery.py`)

This example illustrates how to calculate Landau levels (LLs) in the presence of a perpendicular magnetic field and visualize them as a fan diagram.

The following figure shows a typical LL fan for bilayer graphene:

![Bilayer graphene LL fan diagram](_static/blg_landau_level_fan.png)

**Example: LL fan for bilayer graphene**

For `n_layers=2`, the Bernal and rhombohedral kernels describe the same AB bilayer connectivity. The supported example keeps the user-facing `BernalMultilayer` entry point for consistency with the top-level API.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import contimod_graphene as cg

params = cg.GrapheneTBParameters.preset("blg").replace(U=0.0)
model = cg.BernalMultilayer(n_layers=2, params=params)

B_values = jnp.linspace(0.5, 9.0, 80)

eigvals = []
for B in B_values:
    H = model.landau_level_hamiltonian(float(B), n_cut=40)
    eigvals.append(np.linalg.eigvalsh(np.asarray(H)))

fig, ax = plt.subplots()
for B, e in zip(np.asarray(B_values), eigvals):
    ax.plot(np.full_like(e, B), e, "k.", ms=2)

ax.set_xlabel(r"Magnetic field $B$ [T]")
ax.set_ylabel("Energy [meV]")
ax.set_ylim(-15, 15)
plt.show()
```

## 3. Bernal (ABA) Stacking Analysis (`examples/standalone_gallery.py`)

This example focuses on Bernal (ABA) stacked graphene and demonstrates how to compute zero-field band structures while still going through the same top-level `GrapheneTBParameters` / `BernalMultilayer` surface used elsewhere in the package.

The figure below shows the band structure of ABA trilayer graphene:

![ABA trilayer band structure](_static/aba_trilayer_bandstructure.png)

**Example: zero-field bands for ABA trilayer**

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import contimod_graphene as cg

graphene_params_ABA = {
    "gamma0": 3100,
    "gamma1": 370,
    "gamma2": -19,
    "gamma3": 315,
    "gamma4": 140,
    "gamma5": 20,
    "U": 0.0,
    "Delta": 18.5,
    "delta": 3.8,
}

k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

params_aba = cg.GrapheneTBParameters.from_dict(graphene_params_ABA)
model_aba = cg.BernalMultilayer(n_layers=3, params=params_aba)

Hs = model_aba.hamiltonian_batch(ks)
bands = jnp.linalg.eigvalsh(Hs)

fig, ax = plt.subplots()
for band in bands.T:
    ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)

ax.set_xlabel(r"$k_x\,a$")
ax.set_ylabel("Energy [meV]")
ax.set_ylim(-80, 80)
plt.show()
```

## 4. Explicit Downstream Integration (`contimod_example.ipynb`)

`contimod_example.ipynb` is intentionally the one notebook in this directory that assumes the external `contimod` package is installed. It is a downstream-integration example rather than part of the default `contimod_graphene` usage story.

The older notebooks `bandstructure_plots.ipynb`, `bernal_bands_LL.ipynb`, and `landau_level_fans.ipynb` remain as companion exploratory material. The default package-native workflow, however, should start from `examples/standalone_quickstart.py`, `examples/standalone_gallery.py`, or the snippets in [usage.md](/Users/wolft/Dev/contimod_graphene/docs/usage.md).
