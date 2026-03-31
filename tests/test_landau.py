from __future__ import annotations

import numpy as np

import contimod_graphene as cg
from contimod_graphene.landau import graphene_ll_formfactors, ll_formfactor


def test_ll_formfactor_zero_momentum_is_identity_element():
    assert np.isclose(ll_formfactor(0, 0, 0.0), 1.0)


def test_graphene_ll_formfactors_q0_is_identity_for_asymmetric_blocks():
    ll_block_sizes = np.array([2, 3, 2])
    n_basis = int(np.sum(ll_block_sizes))
    wavefunctions = np.eye(n_basis, dtype=complex)

    formfactors = graphene_ll_formfactors(wavefunctions, ll_block_sizes, np.array([0.0]))

    assert formfactors.shape == (1, n_basis, n_basis)
    np.testing.assert_allclose(formfactors[0], np.eye(n_basis), atol=1e-12, rtol=0.0)


def test_public_api_surface_exports_landau_module():
    assert cg.landau is not None
