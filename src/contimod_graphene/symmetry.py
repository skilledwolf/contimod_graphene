"""Graphene-specific symmetry helpers built on the package basis conventions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

ProjectFn = Callable[[Any], Any]

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
    """C3z action on the Rhombohedral orbital basis.

    The basis ordering used by ``contimod.materials.graphene.Rhombohedral`` is, slowest
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
    (n_fold=3) and feed both into ``hartreefock.symmetry.make_project_fn``'s
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

    The analytic SW / band-restricted reductions of ``Rhombohedral`` retain
    only the outer (A1, B_N) sublattices.  Restricted to that subspace,
    the diagonal C3z phases at valley K reduce to ``ω^0 = 1`` for A1 and
    ``ω^N`` for B_N (from ``ω^((ℓ-1) + s)`` with (ℓ, s) = (1, 0) and
    (N, 1) respectively, using ``ω = exp(2πi/3)``).  At valley K' the K
    block is complex-conjugated; spin acts trivially.

    This is the 2-orbital companion to :func:`make_c3_orbital_unitary`,
    needed by 2-band variants (``Rhombohedral(two_band=True)``,
    ``Rhombohedral(3, two_band=True)``, ``Bernal(2, two_band=True)``) where the full
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


def make_pm_group(
    identity: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
    """Return the standard PM same-k group for spinful, valleyful graphene."""
    identity = np.asarray(identity)
    spin_elems = [identity, np.asarray(s1), np.asarray(s2), np.asarray(s3)]
    valley_elems = [identity, np.asarray(v3)]
    return np.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)


def make_time_reversal_U(
    s2: np.ndarray,
    v1: np.ndarray,
) -> np.ndarray:
    """Return the antiunitary matrix part of graphene time reversal."""
    return np.asarray(v1) @ (1j * np.asarray(s2))


def make_hm_groups(
    identity: np.ndarray,
    s3: np.ndarray,
    v1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the same-k and flip-k groups used by the HM branch."""
    identity = np.asarray(identity)
    s3 = np.asarray(s3)
    v1 = np.asarray(v1)
    same_k = np.stack([identity, s3], axis=0)
    flip_k = np.stack([v1, s3 @ v1], axis=0)
    return same_k, flip_k


def make_svp_symmetry_group(
    *,
    identity: np.ndarray,
    s1: np.ndarray,
    s3: np.ndarray,
    v_rotation: np.ndarray,
    v3: np.ndarray,
    outlier_sv: tuple[int, int] = (+1, +1),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the S3 group that permutes the three inactive spin-valley sectors."""
    identity = np.asarray(identity)
    s1 = np.asarray(s1)
    s3 = np.asarray(s3)
    v_rotation = np.asarray(v_rotation)
    v3 = np.asarray(v3)

    so = float(outlier_sv[0])
    vo = float(outlier_sv[1])

    s1v = s1 @ v_rotation
    s3v3 = s3 @ v3

    T_AB = s1v @ (identity - so * vo * s3v3) / 2 + (identity + so * vo * s3v3) / 2
    T_AC = v_rotation @ (identity - so * s3) / 2 + (identity + so * s3) / 2
    T_BC = s1 @ (identity - vo * v3) / 2 + (identity + vo * v3) / 2

    C1 = T_AB @ T_AC
    C2 = T_AC @ T_AB

    same_k = np.stack([identity, C1, C2], axis=0)
    flip_k = np.stack([T_AB, T_AC, T_BC], axis=0)
    return same_k, flip_k


def make_svp_project_fn(
    *,
    s3: np.ndarray,
    v3: np.ndarray,
    n_orb: int,
    outlier_sv: tuple[int, int] = (+1, +1),
    k_convention: str = "flip",
    k_flip_axes: tuple[int, ...] = (0,),
    k_index_map: tuple[np.ndarray, np.ndarray] | None = None,
    k_valid_mask: np.ndarray | None = None,
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
    mask = mask_np

    k_conv = str(k_convention)
    k_axes = tuple(k_flip_axes)

    if k_index_map is not None:
        k_idx_i = np.asarray(k_index_map[0], dtype=np.int32)
        k_idx_j = np.asarray(k_index_map[1], dtype=np.int32)
    else:
        k_idx_i = k_idx_j = None

    if k_valid_mask is not None:
        k_valid = np.asarray(k_valid_mask, dtype=bool)
    else:
        k_valid = None

    # The projector runs inside the hartreefock solvers; borrow its helpers.
    from hartreefock.symmetry import array_namespace, flip_k

    def _flip_block(xp, A):
        if k_idx_i is not None:
            return A[k_idx_i, k_idx_j]
        return flip_k(xp, A, k_conv, k_axes)

    def _set_blocks(xp, out, blocks):
        if xp is np:
            out = out.copy()
            for sl, value in blocks:
                out[..., sl, sl] = value
            return out
        for sl, value in blocks:
            out = out.at[..., sl, sl].set(value)
        return out

    def project(P):
        xp = array_namespace(P)
        out = P * xp.asarray(mask).astype(P.real.dtype)

        P_same = P[..., sl_same, sl_same]
        P_ov0, P_ov1 = P[..., sl_ov0, sl_ov0], P[..., sl_ov1, sl_ov1]
        Q = (P_same + _flip_block(xp, P_ov0) + _flip_block(xp, P_ov1)) / 3.0
        Q_flip = _flip_block(xp, Q)
        Q_ov0 = Q_ov1 = Q_flip
        if k_valid is not None:
            # Where valid, write the averaged blocks; elsewhere keep the input.
            valid = xp.asarray(k_valid)[..., None, None]
            Q = xp.where(valid, Q, P_same)
            Q_ov0 = xp.where(valid, Q_flip, P_ov0)
            Q_ov1 = xp.where(valid, Q_flip, P_ov1)
        return _set_blocks(xp, out, ((sl_same, Q), (sl_ov0, Q_ov0), (sl_ov1, Q_ov1)))

    return project
