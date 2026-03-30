from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contimod_graphene as cg
from contimod_graphene import bernal, rhombohedral


jax.config.update("jax_enable_x64", True)


def test_public_api_surface_exports_new_model_and_parameter_types():
    assert cg.GrapheneTBParameters is not None
    assert cg.BernalMultilayer is not None
    assert cg.RhombohedralMultilayer is not None
    assert cg.ABAMultilayer is cg.BernalMultilayer
    assert cg.ABCMultilayer is cg.RhombohedralMultilayer


def test_bernal_model_matches_low_level_kernel_and_batch_shape():
    params = cg.GrapheneTBParameters.preset("4lg").replace(U=0.0)
    model = cg.BernalMultilayer(n_layers=4, params=params)

    H_model = model.hamiltonian(0.1, -0.1)
    H_ref = bernal.hamiltonian(0.1, -0.1, n_layers=4, params=params)
    np.testing.assert_allclose(np.array(H_model), np.array(H_ref), atol=1e-8)

    ks = jnp.array([[0.0, 0.0], [0.1, -0.1], [0.2, 0.05]])
    Hs = model.hamiltonian_batch(ks, jit=False)
    assert Hs.shape == (3, 8, 8)
    np.testing.assert_allclose(np.array(Hs[1]), np.array(H_ref), atol=1e-8)


def test_rhombohedral_model_matches_low_level_surfaces():
    params = cg.GrapheneTBParameters.preset("tlg").replace(U=0.0)
    model = cg.RhombohedralMultilayer(n_layers=3, params=params)

    H_model = model.hamiltonian(0.08, 0.02)
    H_ref = rhombohedral.hamiltonian(0.08, 0.02, n_layers=3, params=params)
    np.testing.assert_allclose(np.array(H_model), np.array(H_ref), atol=1e-8)

    H2_model = model.two_band_hamiltonian(0.03, -0.01)
    H2_ref = rhombohedral.hamiltonian_2bands(0.03, -0.01, n_layers=3, params=params)
    np.testing.assert_allclose(np.array(H2_model), np.array(H2_ref), atol=1e-8)

    Hll_model = model.landau_level_hamiltonian(10.0, n_cut=8, valley="K'")
    Hll_ref = rhombohedral.hamiltonian_LL(10.0, n_layers=3, n_cut=8, flip_valley=True, params=params)
    np.testing.assert_allclose(Hll_model, Hll_ref, atol=1e-8)


def test_with_params_returns_new_model_without_mutating_original():
    model = cg.RhombohedralMultilayer()
    shifted = model.with_params(U=11.0)

    assert model.params["U"] != shifted.params["U"]
    assert shifted.params["U"] == pytest.approx(11.0)
    assert shifted.n_layers == model.n_layers


def test_bernal_two_band_is_bilayer_only():
    with pytest.raises(ValueError, match="only defined for bilayer"):
        cg.BernalMultilayer(n_layers=3).two_band_hamiltonian(0.01, 0.02)
