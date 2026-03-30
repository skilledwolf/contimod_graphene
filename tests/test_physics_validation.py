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


def _aba_trilayer_params(*, U: float = 0.0) -> cg.GrapheneTBParameters:
    return cg.GrapheneTBParameters.from_dict(
        {
            "gamma0": 3100.0,
            "gamma1": 370.0,
            "gamma2": -19.0,
            "gamma3": 315.0,
            "gamma4": 140.0,
            "gamma5": 20.0,
            "U": U,
            "Delta": 18.5,
            "delta": 3.8,
        }
    )


def _clean_aba_trilayer_params() -> cg.GrapheneTBParameters:
    return _clean_bernal_bilayer_params()


def _delta_bernal_bilayer_params(Delta: float = 20.0) -> cg.GrapheneTBParameters:
    return _clean_bernal_bilayer_params().replace(Delta=Delta)


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

    mirror_unitary = cg.basis.bernal_trilayer_mirror_unitary()
    mirror_operator = cg.basis.bernal_trilayer_mirror_operator(dtype=float)
    odd_projector, even_projector = cg.basis.bernal_trilayer_mirror_projectors()

    assert mirror_unitary.shape == (6, 6)
    np.testing.assert_allclose(
        mirror_operator,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(mirror_operator @ mirror_operator, np.eye(6), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        mirror_unitary.conj().T @ mirror_unitary,
        np.eye(6),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(mirror_operator, even_projector - odd_projector, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(odd_projector @ odd_projector, odd_projector, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(even_projector @ even_projector, even_projector, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(odd_projector @ even_projector, np.zeros((6, 6)), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(odd_projector + even_projector, np.eye(6), atol=1e-12, rtol=1e-12)


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


def test_blg_delta_splits_low_energy_pair_and_tracks_two_band_asymmetry():
    params = _delta_bernal_bilayer_params()
    model = cg.BernalMultilayer(n_layers=2, params=params)

    full_at_zero = np.linalg.eigvalsh(np.asarray(model.hamiltonian(0.0, 0.0)))
    reduced_at_zero = np.linalg.eigvalsh(np.asarray(model.two_band_hamiltonian(0.0, 0.0)))

    np.testing.assert_allclose(full_at_zero[1:3], np.array([-10.0, 10.0]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(full_at_zero[1:3], reduced_at_zero, atol=1e-10, rtol=1e-10)

    max_rel_err = 0.0
    for k in np.logspace(-5, -2, 9):
        full_low_energy = np.linalg.eigvalsh(np.asarray(model.hamiltonian(float(k), 0.0)))[1:3]
        reduced_evals = np.linalg.eigvalsh(np.asarray(model.two_band_hamiltonian(float(k), 0.0)))
        rel_err = np.max(
            np.abs(full_low_energy - reduced_evals) / np.maximum(np.abs(full_low_energy), 1e-14)
        )
        max_rel_err = max(max_rel_err, float(rel_err))

    assert max_rel_err < 3e-4


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


def test_aba_trilayer_mirror_parity_blocks_decouple_and_u_breaks_them():
    mirror_unitary = cg.basis.bernal_trilayer_mirror_unitary()
    odd_projector, even_projector = cg.basis.bernal_trilayer_mirror_projectors()

    symmetric_model = cg.BernalMultilayer(n_layers=3, params=_aba_trilayer_params(U=0.0))
    biased_model = cg.BernalMultilayer(n_layers=3, params=_aba_trilayer_params(U=30.0))

    for kx, ky in ((0.0, 0.0), (0.01, 0.0), (0.015, -0.02)):
        h_symmetric = np.asarray(symmetric_model.hamiltonian(kx, ky))
        h_mirror = mirror_unitary.conj().T @ h_symmetric @ mirror_unitary
        off_block = h_mirror[:2, 2:]
        scale = max(np.linalg.norm(h_symmetric), 1.0)

        assert np.linalg.norm(off_block) / scale < 1e-12
        np.testing.assert_allclose(
            odd_projector @ h_symmetric @ even_projector,
            np.zeros((6, 6)),
            atol=1e-10,
            rtol=1e-10,
        )

        h_biased = np.asarray(biased_model.hamiltonian(kx, ky))
        biased_coupling = odd_projector @ h_biased @ even_projector
        assert np.linalg.norm(biased_coupling) > 1e-6


def test_aba_trilayer_clean_mirror_sectors_match_monolayer_and_bilayer_spectra():
    params = _clean_aba_trilayer_params()
    model = cg.BernalMultilayer(n_layers=3, params=params)
    mirror_unitary = cg.basis.bernal_trilayer_mirror_unitary()

    velocity = np.sqrt(3.0) * float(params["gamma0"]) / 2.0
    gamma1_eff = np.sqrt(2.0) * float(params["gamma1"])

    for kx, ky in ((1e-5, 0.0), (1e-3, 0.0), (1e-2, 2e-2)):
        h_mirror = mirror_unitary.conj().T @ np.asarray(model.hamiltonian(kx, ky)) @ mirror_unitary

        odd_block = h_mirror[:2, :2]
        even_evals = np.linalg.eigvalsh(h_mirror[2:, 2:])

        pi = kx + 1j * ky
        expected_odd = np.array(
            [
                [0.0, velocity * np.conj(pi)],
                [velocity * pi, 0.0],
            ],
            dtype=complex,
        )
        rho = np.sqrt(gamma1_eff**2 + 4.0 * (velocity * np.abs(pi)) ** 2)
        expected_even = np.array(
            [
                -0.5 * (rho + gamma1_eff),
                -0.5 * (rho - gamma1_eff),
                0.5 * (rho - gamma1_eff),
                0.5 * (rho + gamma1_eff),
            ]
        )

        np.testing.assert_allclose(odd_block, expected_odd, atol=3e-6, rtol=1e-10)
        np.testing.assert_allclose(even_evals, expected_even, atol=2e-6, rtol=1e-10)


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


def test_abc_trilayer_u_opens_gap_while_delta_shifts_low_energy_pair():
    clean_params = _clean_rhombohedral_params("tlg")

    u_model = cg.RhombohedralMultilayer(n_layers=3, params=clean_params.replace(U=12.0, Delta=0.0))
    delta_model = cg.RhombohedralMultilayer(n_layers=3, params=clean_params.replace(U=0.0, Delta=6.0))

    u_full = np.linalg.eigvalsh(np.asarray(u_model.hamiltonian(0.0, 0.0)))
    delta_full = np.linalg.eigvalsh(np.asarray(delta_model.hamiltonian(0.0, 0.0)))
    u_reduced = np.linalg.eigvalsh(np.asarray(u_model.two_band_hamiltonian(0.0, 0.0)))
    delta_reduced = np.linalg.eigvalsh(np.asarray(delta_model.two_band_hamiltonian(0.0, 0.0)))

    np.testing.assert_allclose(u_full[2:4], np.array([-6.0, 6.0]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(u_reduced, np.array([-6.0, 6.0]), atol=1e-10, rtol=1e-10)

    np.testing.assert_allclose(delta_full[2:4], np.array([6.0, 6.0]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(delta_reduced, np.array([6.0, 6.0]), atol=1e-10, rtol=1e-10)


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
