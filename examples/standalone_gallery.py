#!/usr/bin/env python3
"""Standalone public-API examples for contimod_graphene."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import contimod_graphene as cg


DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"


def _prepare_outdir(outdir: Path | str | None) -> Path:
    path = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def abc_trilayer_bandstructure(
    *,
    outdir: Path | str | None = None,
    n_k: int = 400,
    kmax: float = 0.28,
    U: float = 10.0,
) -> dict[str, object]:
    outdir_path = _prepare_outdir(outdir)
    params = cg.GrapheneTBParameters.preset("tlg").replace(U=U)
    model = cg.RhombohedralMultilayer(n_layers=3, params=params)

    k_lin = kmax * jnp.linspace(-0.5, 0.5, int(n_k))
    ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

    hs_full = model.hamiltonian_batch(ks)
    bands_full = jnp.linalg.eigvalsh(hs_full)

    hs_low = jax.vmap(
        lambda kx, ky: model.two_band_hamiltonian(kx, ky),
        in_axes=(0, 0),
    )(*ks.T)
    bands_low = jnp.linalg.eigvalsh(hs_low)

    fig, ax = plt.subplots()
    for band in bands_full.T:
        ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)
    for band in bands_low.T:
        ax.plot(k_lin, band, color="red", linewidth=1.5, linestyle="--", alpha=0.9)

    ax.set_xlabel(r"$k_x\,a$")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-80, 80)

    output_path = outdir_path / "abc_trilayer_bandstructure.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "bands_full_shape": tuple(int(x) for x in bands_full.shape),
        "bands_low_shape": tuple(int(x) for x in bands_low.shape),
    }


def blg_landau_level_fan(
    *,
    outdir: Path | str | None = None,
    n_b: int = 80,
    b_min: float = 0.5,
    b_max: float = 9.0,
    n_cut: int = 40,
) -> dict[str, object]:
    outdir_path = _prepare_outdir(outdir)
    params = cg.GrapheneTBParameters.preset("blg").replace(U=0.0)
    model = cg.BernalMultilayer(n_layers=2, params=params)

    b_values = jnp.linspace(b_min, b_max, int(n_b))
    eigvals = []
    for b in b_values:
        h_ll = model.landau_level_hamiltonian(float(b), n_cut=int(n_cut))
        eigvals.append(np.linalg.eigvalsh(np.asarray(h_ll)))

    fig, ax = plt.subplots()
    for b, energies in zip(np.asarray(b_values), eigvals):
        ax.plot(np.full_like(energies, b), energies, "k.", ms=2)

    ax.set_xlabel(r"Magnetic field $B$ [T]")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-15, 15)

    output_path = outdir_path / "blg_landau_level_fan.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "n_fields": int(n_b),
        "levels_per_field": int(eigvals[0].shape[0]) if eigvals else 0,
    }


def aba_trilayer_bandstructure(
    *,
    outdir: Path | str | None = None,
    n_k: int = 400,
    kmax: float = 0.28,
) -> dict[str, object]:
    outdir_path = _prepare_outdir(outdir)
    graphene_params_aba = {
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

    params = cg.GrapheneTBParameters.from_dict(graphene_params_aba)
    model = cg.BernalMultilayer(n_layers=3, params=params)

    k_lin = kmax * jnp.linspace(-0.5, 0.5, int(n_k))
    ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)
    bands = jnp.linalg.eigvalsh(model.hamiltonian_batch(ks))

    fig, ax = plt.subplots()
    for band in bands.T:
        ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r"$k_x\,a$")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-80, 80)

    output_path = outdir_path / "aba_trilayer_bandstructure.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "bands_shape": tuple(int(x) for x in bands.shape),
    }


def main(
    *,
    outdir: Path | str | None = None,
    n_k: int = 400,
    n_b: int = 80,
    n_cut: int = 40,
) -> dict[str, object]:
    outdir_path = _prepare_outdir(outdir)
    abc = abc_trilayer_bandstructure(outdir=outdir_path, n_k=n_k)
    blg = blg_landau_level_fan(outdir=outdir_path, n_b=n_b, n_cut=n_cut)
    aba = aba_trilayer_bandstructure(outdir=outdir_path, n_k=n_k)

    return {
        "output_paths": [
            abc["output_path"],
            blg["output_path"],
            aba["output_path"],
        ],
        "abc_trilayer": abc,
        "blg_landau_levels": blg,
        "aba_trilayer": aba,
    }


if __name__ == "__main__":
    result = main()
    for path in result["output_paths"]:
        print(f"Saved: {path}")
