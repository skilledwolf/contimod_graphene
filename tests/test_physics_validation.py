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


def _full_rhombohedral_params() -> cg.GrapheneTBParameters:
    return cg.GrapheneTBParameters.preset("4lg").replace(U=0.0, Delta=0.0)


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


def _aba_trilayer_source_aligned_params(*, U: float = 0.0, gamma3: float = 315.0) -> cg.GrapheneTBParameters:
    return _aba_trilayer_params(U=U).replace(Delta=0.0, gamma3=gamma3)


def _clean_aba_trilayer_params() -> cg.GrapheneTBParameters:
    return _clean_bernal_bilayer_params()


def _delta_bernal_bilayer_params(Delta: float = 20.0) -> cg.GrapheneTBParameters:
    return _clean_bernal_bilayer_params().replace(Delta=Delta)


def _full_even_n_bernal_params() -> cg.GrapheneTBParameters:
    return cg.GrapheneTBParameters.preset("4lg").replace(U=0.0, Delta=0.0)


def _aba_trilayer_monolayer_like_params(params: cg.GrapheneTBParameters) -> cg.GrapheneTBParameters:
    gamma2 = float(params["gamma2"])
    gamma5 = float(params["gamma5"])
    return params.replace(
        Delta=float(params["Delta"]) - gamma2,
        delta=float(params["delta"]) - 0.5 * (gamma2 + gamma5),
    )


def _aba_trilayer_bilayer_like_params(params: cg.GrapheneTBParameters) -> cg.GrapheneTBParameters:
    lambda2 = np.sqrt(2.0)
    return params.replace(
        gamma1=lambda2 * float(params["gamma1"]),
        gamma3=lambda2 * float(params["gamma3"]),
        gamma4=lambda2 * float(params["gamma4"]),
        gamma2=0.0,
        gamma5=0.0,
    )


def _aba_trilayer_bilayer_like_w_correction(
    params: cg.GrapheneTBParameters,
    *,
    n_cut: int,
    valley: str,
) -> np.ndarray:
    n_a, n_b = (n_cut - 1, n_cut) if valley == "K" else (n_cut, n_cut - 1)
    zeros_ab = np.zeros((n_a, n_b))
    zeros_ba = np.zeros((n_b, n_a))
    zeros_aa = np.zeros((n_a, n_a))
    zeros_bb = np.zeros((n_b, n_b))

    gamma2 = float(params["gamma2"])
    gamma5 = float(params["gamma5"])

    return np.block(
        [
            [(gamma2 / 2.0) * np.eye(n_a), zeros_ab, zeros_aa, zeros_ab],
            [zeros_ba, (gamma5 / 2.0) * np.eye(n_b), zeros_ba, zeros_bb],
            [zeros_aa, zeros_ab, zeros_aa, zeros_ab],
            [zeros_ba, zeros_bb, zeros_ba, zeros_bb],
        ]
    )


def _aba_trilayer_ll_mirror_blocks(
    model: cg.BernalMultilayer,
    *,
    B: float,
    n_cut: int,
    valley: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_ll = model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)
    d_layer = h_ll.shape[0] // 3
    mirror_block_unitary = cg.basis.bernal_trilayer_mirror_block_unitary(d_layer)
    h_mirror = mirror_block_unitary.conj().T @ h_ll @ mirror_block_unitary
    return h_ll, h_mirror[:d_layer, :d_layer], h_mirror[d_layer:, d_layer:], h_mirror[:d_layer, d_layer:]


def _ll_eigenvalues(
    model: cg.BernalMultilayer | cg.RhombohedralMultilayer,
    *,
    B: float,
    n_cut: int,
    valley: str,
) -> np.ndarray:
    return np.linalg.eigvalsh(np.asarray(model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)))


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
    mirror_layer_unitary = cg.basis.bernal_trilayer_mirror_layer_unitary()
    mirror_block_unitary = cg.basis.bernal_trilayer_mirror_block_unitary(2)
    mirror_operator = cg.basis.bernal_trilayer_mirror_operator(dtype=float)
    odd_projector, even_projector = cg.basis.bernal_trilayer_mirror_projectors()

    assert mirror_unitary.shape == (6, 6)
    np.testing.assert_allclose(mirror_layer_unitary.conj().T @ mirror_layer_unitary, np.eye(3), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mirror_block_unitary, mirror_unitary, atol=1e-12, rtol=1e-12)
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


@pytest.mark.parametrize(
    ("family", "n_layers", "params"),
    [
        pytest.param(
            "bernal",
            2,
            cg.GrapheneTBParameters.preset("blg").replace(U=0.0, Delta=0.0),
            id="bernal-bilayer",
        ),
        pytest.param(
            "bernal",
            4,
            _full_even_n_bernal_params(),
            id="bernal-tetralayer",
        ),
        pytest.param(
            "bernal",
            6,
            _full_even_n_bernal_params(),
            id="bernal-hexalayer",
        ),
        pytest.param("rhombohedral", 3, cg.GrapheneTBParameters.preset("tlg").replace(U=0.0), id="abc-trilayer"),
        pytest.param("rhombohedral", 4, _full_rhombohedral_params(), id="abc-tetralayer"),
        pytest.param("rhombohedral", 5, _full_rhombohedral_params(), id="abc-pentalayer"),
    ],
)
@pytest.mark.parametrize("B", [0.5, 2.0, 5.0])
def test_inversion_symmetric_ll_spectra_match_between_valleys(
    family: str,
    n_layers: int,
    params: cg.GrapheneTBParameters,
    B: float,
):
    if family == "bernal":
        model = cg.BernalMultilayer(n_layers=n_layers, params=params)
    else:
        model = cg.RhombohedralMultilayer(n_layers=n_layers, params=params)

    evals_k = _ll_eigenvalues(model, B=B, n_cut=12, valley="K")
    evals_kprime = _ll_eigenvalues(model, B=B, n_cut=12, valley="K'")

    np.testing.assert_allclose(evals_k, evals_kprime, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("valley", ["K", "K'"])
@pytest.mark.parametrize("B", [0.5, 1.0, 2.0])
def test_aba_trilayer_clean_ll_mirror_blocks_match_monolayer_and_bilayer_like_models(
    B: float,
    valley: str,
):
    params = _clean_aba_trilayer_params()
    trilayer_model = cg.BernalMultilayer(n_layers=3, params=params)
    monolayer_model = cg.BernalMultilayer(n_layers=1, params=params)
    bilayer_like_model = cg.BernalMultilayer(
        n_layers=2,
        params=params.replace(gamma1=np.sqrt(2.0) * float(params["gamma1"])),
    )

    n_cut = 12
    trilayer_ll, odd_block, even_block, off_block = _aba_trilayer_ll_mirror_blocks(
        trilayer_model,
        B=B,
        n_cut=n_cut,
        valley=valley,
    )
    monolayer_ll = monolayer_model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)
    bilayer_like_ll = bilayer_like_model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)

    scale = max(np.linalg.norm(trilayer_ll), 1.0)

    np.testing.assert_allclose(odd_block, monolayer_ll, atol=2e-5, rtol=1e-10)
    np.testing.assert_allclose(even_block, bilayer_like_ll, atol=1e-4, rtol=1e-10)
    assert np.linalg.norm(off_block) / scale < 1e-7


@pytest.mark.parametrize("valley", ["K", "K'"])
@pytest.mark.parametrize("B", [0.5, 1.0, 2.0, 5.0])
def test_aba_trilayer_full_parameter_ll_mirror_blocks_decouple_and_u_breaks_them(
    B: float,
    valley: str,
):
    symmetric_model = cg.BernalMultilayer(n_layers=3, params=_aba_trilayer_params(U=0.0))
    biased_model = cg.BernalMultilayer(n_layers=3, params=_aba_trilayer_params(U=30.0))

    n_cut = 12
    h_ll_sym, odd_block, even_block, off_block = _aba_trilayer_ll_mirror_blocks(
        symmetric_model,
        B=B,
        n_cut=n_cut,
        valley=valley,
    )
    h_ll_biased, _, _, biased_off_block = _aba_trilayer_ll_mirror_blocks(
        biased_model,
        B=B,
        n_cut=n_cut,
        valley=valley,
    )

    sym_scale = max(np.linalg.norm(h_ll_sym), 1.0)
    biased_scale = max(np.linalg.norm(h_ll_biased), 1.0)

    assert np.linalg.norm(off_block) / sym_scale < 1e-7
    np.testing.assert_allclose(
        np.linalg.eigvalsh(h_ll_sym),
        np.sort(np.concatenate([np.linalg.eigvalsh(odd_block), np.linalg.eigvalsh(even_block)])),
        atol=1e-4,
        rtol=1e-10,
    )
    assert np.linalg.norm(biased_off_block) / biased_scale > 1e-3


@pytest.mark.parametrize("valley", ["K", "K'"])
@pytest.mark.parametrize("B", [0.5, 1.0, 2.0, 5.0])
def test_aba_trilayer_full_parameter_odd_ll_block_matches_monolayerlike_block(
    B: float,
    valley: str,
):
    params = _aba_trilayer_params(U=0.0)
    trilayer_model = cg.BernalMultilayer(n_layers=3, params=params)
    monolayerlike_model = cg.BernalMultilayer(
        n_layers=1,
        params=_aba_trilayer_monolayer_like_params(params),
    )

    n_cut = 20
    _, odd_block, _, _ = _aba_trilayer_ll_mirror_blocks(
        trilayer_model,
        B=B,
        n_cut=n_cut,
        valley=valley,
    )
    monolayerlike_ll = monolayerlike_model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)

    np.testing.assert_allclose(odd_block, monolayerlike_ll, atol=5e-5, rtol=1e-10)


@pytest.mark.parametrize("valley", ["K", "K'"])
@pytest.mark.parametrize("B", [0.5, 1.0, 2.0, 5.0])
def test_aba_trilayer_source_aligned_even_ll_block_matches_bilayerlike_h2_block(
    B: float,
    valley: str,
):
    params = _aba_trilayer_source_aligned_params(U=0.0)
    trilayer_model = cg.BernalMultilayer(n_layers=3, params=params)
    bilayerlike_model = cg.BernalMultilayer(
        n_layers=2,
        params=_aba_trilayer_bilayer_like_params(params),
    )

    n_cut = 20
    _, _, even_block, _ = _aba_trilayer_ll_mirror_blocks(
        trilayer_model,
        B=B,
        n_cut=n_cut,
        valley=valley,
    )
    bilayerlike_ll = bilayerlike_model.landau_level_hamiltonian(B, n_cut=n_cut, valley=valley)
    expected_block = bilayerlike_ll + _aba_trilayer_bilayer_like_w_correction(
        params,
        n_cut=n_cut,
        valley=valley,
    )

    np.testing.assert_allclose(even_block, expected_block, atol=5e-5, rtol=1e-10)


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


@pytest.mark.parametrize(("n_layers", "preset"), [(3, "tlg"), (4, "4lg"), (5, "4lg")])
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


@pytest.mark.parametrize(("n_layers", "preset"), [(3, "tlg"), (4, "4lg"), (5, "4lg")])
@pytest.mark.parametrize("valley", ["K", "K'"])
def test_abc_clean_ll_has_n_zero_modes_per_valley(n_layers: int, preset: str, valley: str):
    params = _clean_rhombohedral_params(preset)
    model = cg.RhombohedralMultilayer(n_layers=n_layers, params=params)

    evals = np.linalg.eigvalsh(np.asarray(model.landau_level_hamiltonian(2.0, n_cut=20, valley=valley)))
    abs_evals = np.sort(np.abs(evals))

    assert np.count_nonzero(abs_evals < 1e-10) == n_layers
    assert abs_evals[n_layers] > 1e-4


@pytest.mark.parametrize(("n_layers", "preset", "expected_exponent"), [(3, "tlg", 1.5), (4, "4lg", 2.0)])
def test_abc_clean_ll_lowest_positive_branch_scales_with_expected_field_exponent(
    n_layers: int,
    preset: str,
    expected_exponent: float,
):
    params = _clean_rhombohedral_params(preset)
    b_fields = np.array([0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    slopes = []

    for n_cut in (20, 30):
        model = cg.RhombohedralMultilayer(n_layers=n_layers, params=params)
        lowest_positive = []

        for b_field in b_fields:
            evals = np.linalg.eigvalsh(np.asarray(model.landau_level_hamiltonian(b_field, n_cut=n_cut, valley="K")))
            positive_evals = evals[evals > 1e-10]
            lowest_positive.append(float(positive_evals[0]))

        slopes.append(float(np.polyfit(np.log(b_fields), np.log(lowest_positive), deg=1)[0]))

    assert slopes[0] == pytest.approx(expected_exponent, abs=5e-2)
    assert abs(slopes[1] - slopes[0]) < 1e-3


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
