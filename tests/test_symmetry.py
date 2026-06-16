from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from contimod_graphene.symmetry import (
    make_hm_groups,
    make_pm_group,
    make_svp_project_fn,
    make_svp_symmetry_group,
    make_time_reversal_U,
)


def _flip_k(A: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    return jnp.flip(A, axis=axes)


def _sum_unitary_conj(A: jax.Array, G: jax.Array) -> jax.Array:
    acc = jnp.zeros_like(A)
    for g in G:
        gH = jnp.conj(g.T)
        acc = acc + (g @ A) @ gH
    return acc


def _apply_group_projection(
    A: jax.Array,
    G_same: jax.Array,
    G_flip: jax.Array,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    A_neg = _flip_k(A, flip_axes)
    out = _sum_unitary_conj(A, G_same) + _sum_unitary_conj(A_neg, G_flip)
    return out / float(G_same.shape[0] + G_flip.shape[0])


def _make_spin_valley_ops(nb_per_sector: int = 1):
    nb = 4 * nb_per_sector

    s3_diag = np.array([+1, -1, +1, -1] * nb_per_sector, dtype=np.float32)
    v3_diag = np.array([+1, +1, -1, -1] * nb_per_sector, dtype=np.float32)

    identity = np.eye(nb, dtype=np.complex64)
    s3 = np.diag(s3_diag.astype(np.complex64))
    v3 = np.diag(v3_diag.astype(np.complex64))

    s1 = np.zeros((nb, nb), dtype=np.complex64)
    for base in range(0, nb, 2 * nb_per_sector):
        for i in range(nb_per_sector):
            s1[base + i, base + nb_per_sector + i] = 1.0
            s1[base + nb_per_sector + i, base + i] = 1.0

    v1 = np.zeros((nb, nb), dtype=np.complex64)
    half = 2 * nb_per_sector
    for i in range(half):
        v1[i, i + half] = 1.0
        v1[i + half, i] = 1.0

    return dict(identity=identity, s1=s1, s3=s3, v1=v1, v_rotation=v1, v3=v3)


def _make_svp_project_ops(nb_per_sector: int = 1):
    nb = 4 * nb_per_sector
    s3 = np.diag(np.tile([+1, +1, -1, -1], nb_per_sector).astype(np.complex64))
    v3 = np.diag(np.tile([+1, -1, +1, -1], nb_per_sector).astype(np.complex64))
    return s3, v3


def test_make_pm_group_shape_and_unitarity():
    ops = _make_spin_valley_ops()
    G = make_pm_group(
        ops["identity"], ops["s1"], 1j * ops["s1"] @ ops["s3"], ops["s3"], ops["v3"]
    )

    assert G.shape == (8, 4, 4)
    for g in np.array(G):
        np.testing.assert_allclose(g @ g.conj().T, np.eye(4), atol=1e-6)


def test_make_time_reversal_u_matches_v1_is2():
    ops = _make_spin_valley_ops()
    s2 = 1j * ops["s1"] @ ops["s3"]
    U = make_time_reversal_U(s2, ops["v1"])
    expected = jnp.asarray(ops["v1"]) @ (1j * jnp.asarray(s2))
    np.testing.assert_allclose(np.array(U), np.array(expected), atol=1e-6)


def test_make_hm_groups_returns_expected_elements():
    ops = _make_spin_valley_ops()
    same_k, flip_k = make_hm_groups(ops["identity"], ops["s3"], ops["v1"])

    assert same_k.shape == (2, 4, 4)
    assert flip_k.shape == (2, 4, 4)
    np.testing.assert_allclose(np.array(same_k[0]), ops["identity"], atol=1e-6)
    np.testing.assert_allclose(np.array(same_k[1]), ops["s3"], atol=1e-6)
    np.testing.assert_allclose(np.array(flip_k[0]), ops["v1"], atol=1e-6)
    np.testing.assert_allclose(np.array(flip_k[1]), ops["s3"] @ ops["v1"], atol=1e-6)


def test_svp_group_has_correct_shape():
    ops = _make_spin_valley_ops(nb_per_sector=1)
    same_k, flip_k = make_svp_symmetry_group(
        identity=ops["identity"],
        s1=ops["s1"],
        s3=ops["s3"],
        v_rotation=ops["v_rotation"],
        v3=ops["v3"],
    )

    assert same_k.shape == (3, 4, 4)
    assert flip_k.shape == (3, 4, 4)
    for arr in (same_k, flip_k):
        for g in np.array(arr):
            np.testing.assert_allclose(g @ g.conj().T, np.eye(4), atol=1e-5)


def test_svp_group_permutes_three_sectors_fixes_outlier():
    ops = _make_spin_valley_ops(nb_per_sector=1)
    same_k, flip_k = make_svp_symmetry_group(
        identity=ops["identity"],
        s1=ops["s1"],
        s3=ops["s3"],
        v_rotation=ops["v_rotation"],
        v3=ops["v3"],
        outlier_sv=(+1, +1),
    )

    A = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.complex64))[None, None, ...]
    A_proj = _apply_group_projection(A, same_k, flip_k)
    d = jnp.real(jnp.diag(A_proj[0, 0]))

    assert float(d[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(d[1]) == pytest.approx(3.0, abs=1e-5)
    assert float(d[2]) == pytest.approx(3.0, abs=1e-5)
    assert float(d[3]) == pytest.approx(3.0, abs=1e-5)


def test_svp_group_projection_is_idempotent():
    ops = _make_spin_valley_ops(nb_per_sector=1)
    same_k, flip_k = make_svp_symmetry_group(
        identity=ops["identity"],
        s1=ops["s1"],
        s3=ops["s3"],
        v_rotation=ops["v_rotation"],
        v3=ops["v3"],
    )

    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (4, 4, 4, 4)).astype(jnp.complex64)
    A = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    once = _apply_group_projection(A, same_k, flip_k)
    twice = _apply_group_projection(once, same_k, flip_k)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-5)


def test_svp_project_fn_equalises_inactive_blocks():
    s3, v3 = _make_svp_project_ops(nb_per_sector=1)
    proj = make_svp_project_fn(
        s3=jnp.asarray(s3),
        v3=jnp.asarray(v3),
        n_orb=1,
        outlier_sv=(+1, +1),
        k_convention="flip",
        k_flip_axes=(0,),
    )

    A = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.complex64))[None, None, ...]
    A_proj = proj(A)
    d = jnp.real(jnp.diag(A_proj[0, 0]))

    assert float(d[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(d[1]) == pytest.approx(3.0, abs=1e-5)
    assert float(d[2]) == pytest.approx(3.0, abs=1e-5)
    assert float(d[3]) == pytest.approx(3.0, abs=1e-5)


def test_svp_project_fn_zeros_off_diagonal_blocks():
    s3, v3 = _make_svp_project_ops(nb_per_sector=1)
    proj = make_svp_project_fn(
        s3=jnp.asarray(s3),
        v3=jnp.asarray(v3),
        n_orb=1,
        outlier_sv=(+1, +1),
        k_convention="flip",
        k_flip_axes=(0,),
    )

    A = jnp.ones((1, 1, 4, 4), dtype=jnp.complex64)
    A_proj = proj(A)

    for i in range(4):
        for j in range(4):
            if i != j:
                assert float(jnp.abs(A_proj[0, 0, i, j])) == pytest.approx(0.0, abs=1e-6)


def test_svp_project_fn_does_not_force_k_symmetry_on_outlier():
    s3, v3 = _make_svp_project_ops(nb_per_sector=1)
    proj = make_svp_project_fn(
        s3=jnp.asarray(s3),
        v3=jnp.asarray(v3),
        n_orb=1,
        outlier_sv=(+1, +1),
        k_convention="flip",
        k_flip_axes=(0,),
    )

    nk1, nk2 = 5, 3
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (nk1, nk2, 4, 4)).astype(jnp.complex64)
    A = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    A_proj = proj(A)
    np.testing.assert_allclose(
        np.array(A_proj[..., 0:1, 0:1]),
        np.array(A[..., 0:1, 0:1]),
        atol=1e-7,
    )

    outlier_orig = np.array(A[..., 0, 0])
    outlier_kx_flip = np.flip(outlier_orig, axis=0)
    assert not np.allclose(outlier_orig, outlier_kx_flip, atol=1e-3)


def test_svp_project_fn_multi_orb_idempotent():
    n_orb = 2
    nb = 4 * n_orb
    s3 = np.diag(
        np.array([+1] * n_orb + [+1] * n_orb + [-1] * n_orb + [-1] * n_orb, dtype=np.complex64)
    )
    v3 = np.diag(
        np.array([+1] * n_orb + [-1] * n_orb + [+1] * n_orb + [-1] * n_orb, dtype=np.complex64)
    )
    proj = make_svp_project_fn(
        s3=jnp.asarray(s3),
        v3=jnp.asarray(v3),
        n_orb=n_orb,
        outlier_sv=(+1, +1),
        k_convention="flip",
        k_flip_axes=(0,),
    )

    key = jax.random.PRNGKey(77)
    A = jax.random.normal(key, (5, 5, nb, nb)).astype(jnp.complex64)
    A = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-5)


# ----------------------------------------------------------------------
# C3 orbital unitary
# ----------------------------------------------------------------------

from contimod_graphene.symmetry import (
    C3_OMEGA,
    make_c3_group,
    make_c3_group_2band,
    make_c3_orbital_unitary,
    make_c3_orbital_unitary_2band,
)


def test_make_c3_orbital_unitary_is_cube_root_of_identity():
    """U^3 = I for every (valleyful, spinful) combination."""
    for valleyful in (False, True):
        for spinful in (False, True):
            U = make_c3_orbital_unitary(
                4, valleyful=valleyful, spinful=spinful,
            )
            np.testing.assert_allclose(
                U @ U @ U, np.eye(U.shape[0], dtype=np.complex128),
                atol=1e-12,
                err_msg=f"valleyful={valleyful}, spinful={spinful}",
            )
            np.testing.assert_allclose(
                U @ np.conj(U.T), np.eye(U.shape[0], dtype=np.complex128),
                atol=1e-12,
            )


def test_make_c3_orbital_unitary_block_structure():
    """The K-block diagonal phases are ω^((ℓ-1)+s); K' is the conjugate."""
    N = 4
    U = make_c3_orbital_unitary(N, valleyful=True, spinful=False)
    expected_K = np.array(
        [C3_OMEGA ** ((ell - 1) + s)
         for ell in range(1, N + 1) for s in (0, 1)],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(np.diag(U)[: 2 * N], expected_K, atol=1e-12)
    np.testing.assert_allclose(np.diag(U)[2 * N :], np.conj(expected_K), atol=1e-12)


def test_make_c3_orbital_unitary_shapes():
    """Shape grows as 2 (sublattice) × N (layer) × 2^valleyful × 2^spinful."""
    N = 4
    cases = {
        (False, False): 2 * N,                          # 8
        (True,  False): 2 * 2 * N,                       # 16
        (False, True):  2 * 2 * N,                       # 16
        (True,  True):  2 * 2 * 2 * N,                   # 32
    }
    for (vf, sf), expected_n in cases.items():
        U = make_c3_orbital_unitary(N, valleyful=vf, spinful=sf)
        assert U.shape == (expected_n, expected_n), (vf, sf)


def test_make_c3_orbital_unitary_acts_trivially_on_spin():
    """The spinful unitary is I_2 ⊗ U_valley — it commutes with any spin operator."""
    # ``contimod`` is the private upstream package and is not a dependency of the
    # public release; skip this cross-check when it is unavailable (e.g. CI/PyPI).
    cm = pytest.importorskip("contimod")
    H = cm.graphene.NlayerABC(N=4, valleyful=True, spinful=True, U=0.0)
    U = make_c3_orbital_unitary(4, valleyful=True, spinful=True)
    for axis in (1, 2, 3):
        s = np.asarray(H.spin_op(axis))
        np.testing.assert_allclose(
            U @ s, s @ U, atol=1e-12,
            err_msg=f"C3 should commute with spin_op({axis})",
        )


def test_make_c3_group_returns_three_powers():
    G = make_c3_group(4, valleyful=True, spinful=True)
    assert G.shape == (3, 32, 32)
    np.testing.assert_allclose(G[0], np.eye(32, dtype=np.complex128), atol=1e-12)
    np.testing.assert_allclose(G[1] @ G[1], G[2], atol=1e-12)
    np.testing.assert_allclose(G[2] @ G[1], np.eye(32, dtype=np.complex128), atol=1e-12)


# ---- 2-band (A1, B_N) C3 helpers -----------------------------------------

def test_make_c3_orbital_unitary_2band_phases():
    """In the (A1, B_N) basis the K-block diagonal is (ω^0, ω^N)."""
    for N in (2, 3, 4, 5, 6):
        U = make_c3_orbital_unitary_2band(N, valleyful=False, spinful=False)
        np.testing.assert_allclose(
            np.diag(U), np.array([1.0, C3_OMEGA ** N], dtype=np.complex128),
            atol=1e-12, err_msg=f"N={N}",
        )


def test_make_c3_orbital_unitary_2band_is_cube_root_of_identity():
    for N in (2, 3, 4, 5):
        for valleyful in (False, True):
            for spinful in (False, True):
                U = make_c3_orbital_unitary_2band(
                    N, valleyful=valleyful, spinful=spinful,
                )
                np.testing.assert_allclose(
                    U @ U @ U, np.eye(U.shape[0], dtype=np.complex128),
                    atol=1e-12,
                    err_msg=f"N={N}, valleyful={valleyful}, spinful={spinful}",
                )


def test_make_c3_orbital_unitary_2band_shapes():
    """Shape grows as 2 (sublattice, A1+B_N) × 2^valleyful × 2^spinful."""
    cases = {
        (False, False): 2,
        (True,  False): 4,
        (False, True):  4,
        (True,  True):  8,
    }
    for (vf, sf), expected_n in cases.items():
        U = make_c3_orbital_unitary_2band(4, valleyful=vf, spinful=sf)
        assert U.shape == (expected_n, expected_n), (vf, sf)


def test_make_c3_orbital_unitary_2band_K_prime_conjugate():
    """K' block is the complex conjugate of the K block."""
    N = 4
    U = make_c3_orbital_unitary_2band(N, valleyful=True, spinful=False)
    np.testing.assert_allclose(U[:2, :2], np.diag([1.0, C3_OMEGA ** N]), atol=1e-12)
    np.testing.assert_allclose(
        U[2:, 2:], np.diag([1.0, np.conj(C3_OMEGA ** N)]), atol=1e-12,
    )
    # No off-diagonal valley mixing.
    np.testing.assert_allclose(U[:2, 2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(U[2:, :2], 0.0, atol=1e-12)


def test_make_c3_group_2band_shape_and_powers():
    G = make_c3_group_2band(4, valleyful=True, spinful=True)
    assert G.shape == (3, 8, 8)
    np.testing.assert_allclose(G[0], np.eye(8, dtype=np.complex128), atol=1e-12)
    np.testing.assert_allclose(G[1] @ G[1], G[2], atol=1e-12)
    np.testing.assert_allclose(G[2] @ G[1], np.eye(8, dtype=np.complex128), atol=1e-12)


def test_make_c3_orbital_unitary_2band_rejects_n_layers_lt_2():
    with pytest.raises(ValueError, match="n_layers >= 2"):
        make_c3_orbital_unitary_2band(1)


def test_make_c3_orbital_unitary_2band_matches_full_at_active_indices():
    """The 2-band C3 phases should equal the corresponding entries of the full
    (A1, B1, ..., A_N, B_N) C3 unitary at indices 0 (A1) and 2N-1 (B_N)."""
    for N in (2, 3, 4, 5):
        U_full = make_c3_orbital_unitary(N, valleyful=False, spinful=False)
        U_2b = make_c3_orbital_unitary_2band(N, valleyful=False, spinful=False)
        np.testing.assert_allclose(
            U_2b[0, 0], U_full[0, 0], atol=1e-12, err_msg=f"N={N} A1",
        )
        np.testing.assert_allclose(
            U_2b[1, 1], U_full[2 * N - 1, 2 * N - 1], atol=1e-12,
            err_msg=f"N={N} B_N",
        )
