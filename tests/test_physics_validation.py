from __future__ import annotations

import jax
import numpy as np
import pytest

import contimod_graphene as cg


jax.config.update("jax_enable_x64", True)


def _clean_bernal_bilayer_params() -> cg.GrapheneTBParameters:
    return cg.GrapheneTBParameters.preset("blg").replace(
        gamma2=0.0,
        gamma3=0.0,
        gamma4=0.0,
        gamma5=0.0,
        U=0.0,
        Delta=0.0,
        delta=0.0,
    )


def _clean_rhombohedral_params(preset: str) -> cg.GrapheneTBParameters:
    return cg.GrapheneTBParameters.preset(preset).replace(
        gamma2=0.0,
        gamma3=0.0,
        gamma4=0.0,
        gamma5=0.0,
        U=0.0,
        Delta=0.0,
        delta=0.0,
    )


def test_zero_field_basis_helpers_expose_named_site_metadata():
    assert cg.basis.zero_field_orbital_labels(4) == ("A1", "B1", "A2", "B2", "A3", "B3", "A4", "B4")
    assert cg.basis.zero_field_orbital_index(4, 3, "B") == 5
    assert cg.basis.rhombohedral_outer_site_indices(4) == (0, 7)

    expected_layer_mask = np.array([False, False, True, True, False, False, True, True])
    np.testing.assert_array_equal(
        cg.basis.zero_field_orbital_mask(4, layer=(2, 4)),
        expected_layer_mask,
    )

    expected_named_mask = np.array([True, False, False, False, True, False, False, False])
    np.testing.assert_array_equal(
        cg.basis.zero_field_orbital_mask(4, layer=(1, 3), sublattice="A"),
        expected_named_mask,
    )

    projector = cg.basis.zero_field_orbital_projector(4, layer=4, sublattice="B", dtype=float)
    np.testing.assert_array_equal(np.diag(projector), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    np.testing.assert_array_equal(
        cg.basis.bernal_nondimer_mask(4),
        np.array([True, False, False, True, True, False, False, True]),
    )
    np.testing.assert_array_equal(
        cg.basis.bernal_dimer_mask(4),
        np.array([False, True, True, False, False, True, True, False]),
    )


def test_bilayer_zero_field_spectra_match_between_bernal_and_rhombohedral_models():
    params = _clean_bernal_bilayer_params()
    bernal_model = cg.BernalMultilayer(n_layers=2, params=params)
    rhombohedral_model = cg.RhombohedralMultilayer(n_layers=2, params=params)

    for kx, ky in ((0.0, 0.0), (0.01, 0.02), (0.03, -0.01)):
        bernal_evals = np.linalg.eigvalsh(np.asarray(bernal_model.hamiltonian(kx, ky)))
        rhombohedral_evals = np.linalg.eigvalsh(np.asarray(rhombohedral_model.hamiltonian(kx, ky)))
        np.testing.assert_allclose(bernal_evals, rhombohedral_evals, atol=1e-10, rtol=1e-10)


def test_blg_zero_field_exact_bands_in_clean_subspace():
    params = _clean_bernal_bilayer_params()
    model = cg.BernalMultilayer(n_layers=2, params=params)

    gamma1 = float(params["gamma1"])
    velocity = np.sqrt(3.0) * float(params["gamma0"]) / 2.0

    for k in np.logspace(-5, -1, 12):
        rho = np.sqrt(gamma1**2 + 4.0 * (velocity * k) ** 2)
        expected = np.array(
            [
                -0.5 * (rho + gamma1),
                -0.5 * (rho - gamma1),
                0.5 * (rho - gamma1),
                0.5 * (rho + gamma1),
            ]
        )
        evals = np.linalg.eigvalsh(np.asarray(model.hamiltonian(float(k), 0.0)))
        np.testing.assert_allclose(evals, expected, atol=1e-12, rtol=1e-10)


def test_blg_two_band_matches_full_low_energy_branch():
    params = _clean_bernal_bilayer_params()
    model = cg.BernalMultilayer(n_layers=2, params=params)

    max_rel_err = 0.0
    for k in np.logspace(-5, -2.3, 10):
        full_evals = np.linalg.eigvalsh(np.asarray(model.hamiltonian(float(k), 0.0)))
        reduced_evals = np.linalg.eigvalsh(np.asarray(model.two_band_hamiltonian(float(k), 0.0)))

        full_low_energy = full_evals[1:3]
        rel_err = np.max(
            np.abs(full_low_energy - reduced_evals) / np.maximum(np.abs(full_low_energy), 1e-14)
        )
        max_rel_err = max(max_rel_err, float(rel_err))

    assert max_rel_err < 2e-3


@pytest.mark.parametrize("valley", ["K", "K'"])
@pytest.mark.parametrize("B", [0.5, 1.0, 2.0])
def test_blg_ll_zero_modes_and_ab_equivalence(B: float, valley: str):
    params = _clean_bernal_bilayer_params()
    bernal_model = cg.BernalMultilayer(n_layers=2, params=params)
    rhombohedral_model = cg.RhombohedralMultilayer(n_layers=2, params=params)

    bernal_evals = np.linalg.eigvalsh(bernal_model.landau_level_hamiltonian(B, n_cut=40, valley=valley))
    rhombohedral_evals = np.linalg.eigvalsh(rhombohedral_model.landau_level_hamiltonian(B, n_cut=40, valley=valley))

    np.testing.assert_allclose(bernal_evals, rhombohedral_evals, atol=5e-5, rtol=1e-10)
    assert np.count_nonzero(np.abs(bernal_evals) < 1e-10) == 2


@pytest.mark.parametrize(("n_layers", "preset"), [(3, "tlg"), (4, "4lg")])
def test_abc_outer_site_zero_modes_at_k0(n_layers: int, preset: str):
    params = _clean_rhombohedral_params(preset)
    model = cg.RhombohedralMultilayer(n_layers=n_layers, params=params)

    evals, eigenvectors = np.linalg.eigh(np.asarray(model.hamiltonian(0.0, 0.0)))
    zero_mode_indices = np.flatnonzero(np.abs(evals) < 1e-12)
    assert len(zero_mode_indices) == 2

    outer_projector = (
        cg.basis.zero_field_orbital_projector(n_layers, layer=1, sublattice="A")
        + cg.basis.zero_field_orbital_projector(n_layers, layer=n_layers, sublattice="B")
    )

    for index in zero_mode_indices:
        state = eigenvectors[:, index]
        outer_weight = np.real_if_close(state.conj() @ outer_projector @ state)
        assert float(outer_weight) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(("n_layers", "preset", "expected_power"), [(3, "tlg", 3.0), (4, "4lg", 4.0)])
def test_abc_low_energy_scaling_and_two_band_agreement(
    n_layers: int,
    preset: str,
    expected_power: float,
):
    params = _clean_rhombohedral_params(preset)
    model = cg.RhombohedralMultilayer(n_layers=n_layers, params=params)

    ks = np.logspace(-5, -2, 8)
    full_low_band = []
    reduced_low_band = []

    for k in ks:
        full_evals = np.linalg.eigvalsh(np.asarray(model.hamiltonian(float(k), 0.0)))
        reduced_evals = np.linalg.eigvalsh(np.asarray(model.two_band_hamiltonian(float(k), 0.0)))
        full_low_band.append(float(full_evals[n_layers]))
        reduced_low_band.append(float(reduced_evals[1]))

    full_low_band = np.asarray(full_low_band)
    reduced_low_band = np.asarray(reduced_low_band)

    slope = np.polyfit(np.log(ks), np.log(full_low_band), deg=1)[0]
    max_rel_err = np.max(np.abs(full_low_band - reduced_low_band) / full_low_band)

    assert slope == pytest.approx(expected_power, abs=1e-2)
    assert max_rel_err < 5e-3
