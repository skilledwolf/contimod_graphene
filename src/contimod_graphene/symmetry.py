"""Graphene-specific symmetry helpers built on the package basis conventions."""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


ProjectFn = Callable[[jax.Array], jax.Array]

__all__ = [
    "ProjectFn",
    "make_c3_group",
    "make_c3_group_2band",
    "make_c3_orbital_unitary",
    "make_c3_orbital_unitary_2band",
    "make_hm_groups",
    "make_pm_group",
    "make_svp_project_fn",
    "make_svp_symmetry_group",
    "make_time_reversal_U",
]


C3_OMEGA = np.exp(2.0j * np.pi / 3.0)


def make_c3_orbital_unitary(
    n_layers: int,
    *,
    valleyful: bool = True,
    spinful: bool = True,
) -> np.ndarray:
    """C3z action on the NlayerABC orbital basis.

    The basis ordering used by ``contimod_graphene.NlayerABC`` is, slowest
    axis first: ``spin`` (when ``spinful``), ``valley`` (when ``valleyful``),
    then the eight ``(layer, sublattice)`` orbitals ``A1, B1, A2, B2, ..., AN, BN``.

    The C3z unitary is diagonal:

    - At valley ``K``  : ``diag(ω^((ℓ-1) + s))`` for ``ℓ ∈ 1..N``, ``s ∈ {0, 1}``
      (sublattice index ``A=0``, ``B=1``), with ``ω = exp(2πi/3)``.
    - At valley ``K'`` : the complex conjugate of the K block.
    - In spin space   : the identity (C3z does not act on physical spin in the
      spinless k·p limit used here).

    Parameters
    ----------
    n_layers : int
        Number of layers ``N``.
    valleyful : bool
        If ``True``, the returned matrix is block-diagonal in valley with
        ``U_K`` and ``U_K* ``.  If ``False``, only the K block is returned.
    spinful : bool
        If ``True``, the matrix is wrapped in an ``I_2 ⊗ U_valley`` Kron
        product so it acts trivially on spin.

    Returns
    -------
    U : ndarray, complex128, shape ``(nb, nb)``
        The orbital action of one C3 step (i.e. ``U^3 = I``).  Suitable for
        use with ``make_project_fn(spatial_group=stack([U^0, U^1, U^2], 0), ...)``.
    """
    N = int(n_layers)
    if N < 1:
        raise ValueError(f"n_layers must be a positive integer; got {n_layers!r}.")
    diag_K = np.array(
        [C3_OMEGA ** ((ell - 1) + s) for ell in range(1, N + 1) for s in (0, 1)],
        dtype=np.complex128,
    )
    U_K = np.diag(diag_K)
    if valleyful:
        U_Kprime = np.conj(U_K)
        n_per_v = 2 * N
        U_valley = np.zeros((2 * n_per_v, 2 * n_per_v), dtype=np.complex128)
        U_valley[:n_per_v, :n_per_v] = U_K
        U_valley[n_per_v:, n_per_v:] = U_Kprime
    else:
        U_valley = U_K
    if spinful:
        return np.kron(np.eye(2, dtype=np.complex128), U_valley)
    return np.ascontiguousarray(U_valley)


def make_c3_group(
    n_layers: int,
    *,
    valleyful: bool = True,
    spinful: bool = True,
) -> np.ndarray:
    """Stack ``[I, U, U²]`` for the orbital action of the C3 cyclic group.

    Pair this with ``contimod.symmetry.continuum.cyclic_rotation_index_maps``
    (n_fold=3) and feed both into ``jax_hf.symmetry.make_project_fn``'s
    ``spatial_group`` + ``spatial_k_index_maps`` arguments to get the
    continuum-patch C3 projector.
    """
    U = make_c3_orbital_unitary(
        n_layers, valleyful=valleyful, spinful=spinful,
    )
    nb = U.shape[0]
    I = np.eye(nb, dtype=np.complex128)
    U2 = U @ U
    return np.stack([I, U, U2], axis=0)


def make_c3_orbital_unitary_2band(
    n_layers: int,
    *,
    valleyful: bool = True,
    spinful: bool = True,
) -> np.ndarray:
    """C3z action on the *projected* 2-band ABC basis (A1, B_N).

    The analytic SW / band-restricted reductions of ``NlayerABC`` retain
    only the outer (A1, B_N) sublattices.  Restricted to that subspace,
    the diagonal C3z phases at valley K reduce to ``ω^0 = 1`` for A1 and
    ``ω^N`` for B_N (from ``ω^((ℓ-1) + s)`` with (ℓ, s) = (1, 0) and
    (N, 1) respectively, using ``ω = exp(2πi/3)``).  At valley K' the K
    block is complex-conjugated; spin acts trivially.

    This is the 2-orbital companion to :func:`make_c3_orbital_unitary`,
    needed by 2-band variants (``NlayerABC_2bands``,
    ``MultilayerABC_2bands``, ``MultilayerAB_2bands``) where the full
    N-layer C3 unitary would have the wrong dimension.

    Parameters
    ----------
    n_layers : int
        Number of layers ``N`` in the underlying ABC stack.
    valleyful, spinful :
        Same semantics as :func:`make_c3_orbital_unitary`.

    Returns
    -------
    U : ndarray, complex128
        ``2 × 2`` (no flavors) up to ``8 × 8`` (spinful + valleyful) C3
        orbital unitary in the (A1, B_N) basis.
    """
    N = int(n_layers)
    if N < 2:
        raise ValueError(
            f"2-band C3 requires n_layers >= 2 (need a B_N distinct from B1); got {n_layers!r}."
        )
    diag_K = np.array([1.0, C3_OMEGA ** N], dtype=np.complex128)
    U_K = np.diag(diag_K)
    if valleyful:
        U_Kprime = np.conj(U_K)
        U_valley = np.zeros((4, 4), dtype=np.complex128)
        U_valley[:2, :2] = U_K
        U_valley[2:, 2:] = U_Kprime
    else:
        U_valley = U_K
    if spinful:
        return np.kron(np.eye(2, dtype=np.complex128), U_valley)
    return np.ascontiguousarray(U_valley)


def make_c3_group_2band(
    n_layers: int,
    *,
    valleyful: bool = True,
    spinful: bool = True,
) -> np.ndarray:
    """``[I, U, U²]`` cyclic group for the 2-band (A1, B_N) C3 unitary.

    2-band counterpart to :func:`make_c3_group`; pairs with
    ``contimod.symmetry.continuum.cyclic_rotation_index_maps`` exactly
    as the full-basis version does.
    """
    U = make_c3_orbital_unitary_2band(
        n_layers, valleyful=valleyful, spinful=spinful,
    )
    nb = U.shape[0]
    I = np.eye(nb, dtype=np.complex128)
    U2 = U @ U
    return np.stack([I, U, U2], axis=0)


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
    k_index_map: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array] | None = None,
    k_valid_mask: np.ndarray | jax.Array | None = None,
) -> ProjectFn:
    """Build the custom SVP projector that leaves the outlier block untouched.

    Parameters
    ----------
    s3, v3 :
        Spin and valley diagonal operators (used to identify the four
        ``(s, v)`` sectors).
    n_orb :
        Number of orbitals per ``(s, v)`` block.
    outlier_sv :
        ``(s, v)`` of the outlier sector to leave untouched.
    k_convention, k_flip_axes :
        Periodic-grid mirror convention used when ``k_index_map`` is not
        supplied.  ``"flip"`` reflects the chosen axes; ``"mod"`` is the
        ``-k mod nk`` wrap.
    k_index_map :
        Optional ``(idx_i, idx_j)`` tuple of integer arrays of shape
        ``(nk1, nk2)`` giving the index of the kx-mirror partner for each
        k.  When supplied, this overrides the periodic ``k_convention`` —
        useful for finite k·p continuum patches where ``-k`` does not
        always lie inside the patch.
    k_valid_mask :
        Optional boolean mask of shape ``(nk1, nk2)``: where ``False``,
        the SVP averaging is skipped (the outlier-block mask is still
        applied so cross-block elements are zeroed out).
    """
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

    if k_index_map is not None:
        k_idx_i = jnp.asarray(k_index_map[0], dtype=jnp.int32)
        k_idx_j = jnp.asarray(k_index_map[1], dtype=jnp.int32)
    else:
        k_idx_i = k_idx_j = None

    if k_valid_mask is not None:
        k_valid = jnp.asarray(k_valid_mask, dtype=bool)
    else:
        k_valid = None

    def _flip_block(A: jax.Array) -> jax.Array:
        if k_idx_i is not None:
            return A[k_idx_i, k_idx_j]
        return _flip_k(A, k_conv, k_axes)

    def project(P: jax.Array) -> jax.Array:
        out = P * mask

        P_same = P[..., sl_same, sl_same]
        P_ov0_flip = _flip_block(P[..., sl_ov0, sl_ov0])
        P_ov1_flip = _flip_block(P[..., sl_ov1, sl_ov1])

        Q = (P_same + P_ov0_flip + P_ov1_flip) / 3.0
        Q_flip = _flip_block(Q)

        if k_valid is not None:
            # Where valid: write averaged blocks; where invalid: keep input.
            mask_b = k_valid[..., None, None]
            same_in = P[..., sl_same, sl_same]
            ov0_in = P[..., sl_ov0, sl_ov0]
            ov1_in = P[..., sl_ov1, sl_ov1]
            Q_w = jnp.where(mask_b, Q, same_in)
            Q_flip_ov0 = jnp.where(mask_b, Q_flip, ov0_in)
            Q_flip_ov1 = jnp.where(mask_b, Q_flip, ov1_in)
            out = out.at[..., sl_same, sl_same].set(Q_w)
            out = out.at[..., sl_ov0, sl_ov0].set(Q_flip_ov0)
            out = out.at[..., sl_ov1, sl_ov1].set(Q_flip_ov1)
        else:
            out = out.at[..., sl_same, sl_same].set(Q)
            out = out.at[..., sl_ov0, sl_ov0].set(Q_flip)
            out = out.at[..., sl_ov1, sl_ov1].set(Q_flip)
        return out

    return project
