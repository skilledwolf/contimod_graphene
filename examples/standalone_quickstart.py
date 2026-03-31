#!/usr/bin/env python3
"""Minimal standalone quickstart for the contimod_graphene public API."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import contimod_graphene as cg


DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"


def main(
    *,
    outdir: str | Path | None = None,
    num_k: int = 41,
    ll_n_cut: int = 8,
) -> dict[str, object]:
    outdir = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    # Pin Delta=0 here so the quickstart isolates the outer-layer bias U.
    params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0, Delta=0.0)
    abc = cg.RhombohedralMultilayer(n_layers=3, params=params)
    ab = cg.BernalMultilayer(n_layers=2)

    h0 = abc.hamiltonian(0.1, 0.0)
    h2 = abc.two_band_hamiltonian(0.1, 0.0)
    sample_h0 = np.linalg.eigvalsh(np.asarray(abc.hamiltonian(0.02, 0.0)))
    sample_h2 = np.linalg.eigvalsh(np.asarray(abc.two_band_hamiltonian(0.02, 0.0)))

    k_lin = 0.28 * jnp.linspace(-0.5, 0.5, int(num_k))
    ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)
    bands = jnp.linalg.eigvalsh(abc.hamiltonian_batch(ks))

    h_ll = ab.landau_level_hamiltonian(10.0, n_cut=int(ll_n_cut), valley="K")
    ll_eigs = np.linalg.eigvalsh(np.asarray(h_ll))
    ll_near_zero = np.sort(ll_eigs[np.argsort(np.abs(ll_eigs))[: min(8, ll_eigs.size)]])

    summary = {
        "parameter_preset": params.preset_name,
        "available_presets": cg.list_parameter_sets(),
        "abc_u_mev": float(params["U"]),
        "abc_delta_layer_offset_mev": float(params["Delta"]),
        "sample_k_point": [0.02, 0.0],
        "sample_zero_field_eigs_mev": [round(float(x), 3) for x in sample_h0],
        "sample_two_band_eigs_mev": [round(float(x), 3) for x in sample_h2],
        "zero_field_shape": list(np.asarray(h0).shape),
        "two_band_shape": list(np.asarray(h2).shape),
        "band_shape": list(np.asarray(bands).shape),
        "landau_level_shape": list(np.asarray(h_ll).shape),
        "sample_ll_near_zero_mev": [round(float(x), 3) for x in ll_near_zero],
        "band_extrema_mev": [
            round(float(np.min(np.asarray(bands))), 3),
            round(float(np.max(np.asarray(bands))), 3),
        ],
    }

    summary_path = outdir / "standalone_quickstart_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    print(f"Saved: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
