"""Graphene-specific symmetry helpers built on the package basis conventions."""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


ProjectFn = Callable[[jax.Array], jax.Array]

__all__ = [
    "ProjectFn",
    "make_hm_groups",
    "make_pm_group",
    "make_svp_project_fn",
    "make_svp_symmetry_group",
    "make_time_reversal_U",
]


def _flip_k(
    A: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    if k_convention == "flip":
        return jnp.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
            if 0 in flip_axes
            else jnp.arange(nk1, dtype=jnp.int32)
        )
        j = (
            (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
            if 1 in flip_axes
            else jnp.arange(nk2, dtype=jnp.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


def make_pm_group(
    identity: np.ndarray | jax.Array,
    s1: np.ndarray | jax.Array,
    s2: np.ndarray | jax.Array,
    s3: np.ndarray | jax.Array,
    v3: np.ndarray | jax.Array,
) -> jax.Array:
    """Return the standard PM same-k group for spinful, valleyful graphene."""
    identity = jnp.asarray(identity)
    spin_elems = [identity, jnp.asarray(s1), jnp.asarray(s2), jnp.asarray(s3)]
    valley_elems = [identity, jnp.asarray(v3)]
    return jnp.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)


def make_time_reversal_U(
    s2: np.ndarray | jax.Array,
    v1: np.ndarray | jax.Array,
) -> jax.Array:
    """Return the antiunitary matrix part of graphene time reversal."""
    return jnp.asarray(v1) @ (1j * jnp.asarray(s2))


def make_hm_groups(
    identity: np.ndarray | jax.Array,
    s3: np.ndarray | jax.Array,
    v1: np.ndarray | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the same-k and flip-k groups used by the HM branch."""
    identity = jnp.asarray(identity)
    s3 = jnp.asarray(s3)
    v1 = jnp.asarray(v1)
    same_k = jnp.stack([identity, s3], axis=0)
    flip_k = jnp.stack([v1, s3 @ v1], axis=0)
    return same_k, flip_k


def make_svp_symmetry_group(
    *,
    identity: np.ndarray | jax.Array,
    s1: np.ndarray | jax.Array,
    s3: np.ndarray | jax.Array,
    v_rotation: np.ndarray | jax.Array,
    v3: np.ndarray | jax.Array,
    outlier_sv: tuple[int, int] = (+1, +1),
) -> tuple[jax.Array, jax.Array]:
    """Build the S3 group that permutes the three inactive spin-valley sectors."""
    identity = jnp.asarray(identity)
    s1 = jnp.asarray(s1)
    s3 = jnp.asarray(s3)
    v_rotation = jnp.asarray(v_rotation)
    v3 = jnp.asarray(v3)

    so = float(outlier_sv[0])
    vo = float(outlier_sv[1])

    s1v = s1 @ v_rotation
    s3v3 = s3 @ v3

    T_AB = s1v @ (identity - so * vo * s3v3) / 2 + (identity + so * vo * s3v3) / 2
    T_AC = v_rotation @ (identity - so * s3) / 2 + (identity + so * s3) / 2
    T_BC = s1 @ (identity - vo * v3) / 2 + (identity + vo * v3) / 2

    C1 = T_AB @ T_AC
    C2 = T_AC @ T_AB

    same_k = jnp.stack([identity, C1, C2], axis=0)
    flip_k = jnp.stack([T_AB, T_AC, T_BC], axis=0)
    return same_k, flip_k


def make_svp_project_fn(
    *,
    s3: np.ndarray | jax.Array,
    v3: np.ndarray | jax.Array,
    n_orb: int,
    outlier_sv: tuple[int, int] = (+1, +1),
    k_convention: str = "flip",
    k_flip_axes: tuple[int, ...] = (0,),
) -> ProjectFn:
    """Build the custom SVP projector that leaves the outlier block untouched."""
    s3_np = np.asarray(s3)
    v3_np = np.asarray(v3)
    nb = s3_np.shape[0]
    n_blocks = nb // n_orb
    so, vo = float(outlier_sv[0]), float(outlier_sv[1])

    idx_outlier = None
    idx_same_v = None
    idx_other_v: list[int] = []

    for i in range(n_blocks):
        s_val = float(np.sign(np.real(s3_np[i * n_orb, i * n_orb])))
        v_val = float(np.sign(np.real(v3_np[i * n_orb, i * n_orb])))
        if s_val == so and v_val == vo:
            idx_outlier = i
        elif v_val == vo:
            idx_same_v = i
        else:
            idx_other_v.append(i)

    if idx_outlier is None or idx_same_v is None or len(idx_other_v) != 2:
        raise ValueError(
            "Could not identify 4 spin-valley blocks from s3/v3 "
            f"(n_orb={n_orb}, nb={nb}, outlier_sv={outlier_sv})"
        )

    def _sl(i: int) -> slice:
        return slice(i * n_orb, (i + 1) * n_orb)

    sl_same = _sl(idx_same_v)
    sl_ov0 = _sl(idx_other_v[0])
    sl_ov1 = _sl(idx_other_v[1])

    mask_np = np.zeros((nb, nb), dtype=np.float32)
    for i in range(n_blocks):
        a, b = i * n_orb, (i + 1) * n_orb
        mask_np[a:b, a:b] = 1.0
    mask = jnp.asarray(mask_np)

    k_conv = str(k_convention)
    k_axes = tuple(k_flip_axes)

    def project(P: jax.Array) -> jax.Array:
        out = P * mask

        P_same = P[..., sl_same, sl_same]
        P_ov0_flip = _flip_k(P[..., sl_ov0, sl_ov0], k_conv, k_axes)
        P_ov1_flip = _flip_k(P[..., sl_ov1, sl_ov1], k_conv, k_axes)

        Q = (P_same + P_ov0_flip + P_ov1_flip) / 3.0
        Q_flip = _flip_k(Q, k_conv, k_axes)

        out = out.at[..., sl_same, sl_same].set(Q)
        out = out.at[..., sl_ov0, sl_ov0].set(Q_flip)
        out = out.at[..., sl_ov1, sl_ov1].set(Q_flip)
        return out

    return project
