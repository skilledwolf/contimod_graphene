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
