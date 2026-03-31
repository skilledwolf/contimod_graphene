# Physics Validation Plan

This note records the package-specific convention map and the first paper-backed validation wave for `contimod_graphene`. It is intentionally narrower than a literature review: the goal is to explain which claims the test suite asserts today, which paper results those tests come from, and which physics checks are still deferred.

## Package Conventions

- Zero-field full Hamiltonians use the orbital ordering `(A1, B1, A2, B2, ..., AN, BN)`.
- The zero-field public API is valley-fixed. Tests at zero field therefore compare spectra and low-energy structure within that fixed convention rather than asserting explicit `K/K'` degeneracies.
- Landau-level Hamiltonians use asymmetric LL bases that depend on the selected valley. LL tests compare spectra only, not raw matrix elements or basis vectors.
- In the Bernal family, `Delta` is implemented as a full-model A/B sublattice asymmetry `(+Delta/2 on A, -Delta/2 on B)`, while `delta` remains the dimer versus non-dimer onsite offset and `U` remains the layer-asymmetry term.
- For even-layer Bernal valley-degeneracy checks, package `Delta` must vanish: inversion swaps `A_j <-> B_{N+1-j}`, so this package `Delta` term is inversion-odd even though `delta`, `gamma2`, `gamma3`, `gamma4`, and `gamma5` are allowed by the symmetry argument.
- In the rhombohedral family, `U` remains the layer-asymmetry / outer-layer bias term. `Delta` now matches the trilayer `Δ2` convention for `N=3` and is extended for `N>3` as a single inversion-even layer-curvature profile; that `N>3` extension is a package inference rather than a direct paper convention.
- First-wave analytic checks use explicit clean-subspace parameter overlays instead of package defaults:
  - BLG clean subset: `gamma2=gamma3=gamma4=gamma5=U=Delta=delta=0`
  - ABC clean subset (`N=3,4`): `gamma2=gamma3=gamma4=gamma5=U=Delta=delta=0`
- Tests compare sorted eigenvalues, named-site projector weights, or fitted low-energy exponents. They do not compare raw matrices against paper conventions unless the basis is exposed explicitly.

## Literature Anchors

| Reference | Scope in this repo | Current use |
| --- | --- | --- |
| McCann & Koshino, Rep. Prog. Phys. 76, 056503 (2013) | AB bilayer zero-field 4-band spectrum and low-energy two-band reduction | Anchors the clean BLG exact-band and low-energy reduction tests |
| Koshino & McCann, Phys. Rev. B 83, 165443 (2011) | ABA trilayer mirror-parity decomposition and ABAB four-layer next-nearest-layer structure | Anchors the ABA trilayer mirror-parity block-decoupling tests, the clean odd/even sector spectrum checks, the clean LL decomposition into monolayer-like and bilayer-like sectors, the full-parameter LL mirror-decoupling checks, and the ABAB four-layer `W/W'` next-nearest-layer parity test |
| Koshino & McCann, Phys. Rev. B 80, 165409 (2009) | ABC trilayer and general `N`-layer low-energy reductions | Anchors the ABC `k^3` / `k^4` scaling and full-vs-two-band checks |
| Koshino & McCann, Phys. Rev. B 81, 115315 (2010) | Inversion-protected valley degeneracy in even Bernal stacks and all rhombohedral stacks | Anchors the LL `K/K'` spectral-degeneracy checks that are currently asserted for Bernal bilayer/tetralayer/hexalayer at `U=Delta=0` and rhombohedral trilayer/tetralayer/pentalayer at `U=0` |
| Min & MacDonald, Prog. Theor. Phys. Suppl. 176, 227 (2008) | Nearest-neighbor chiral doublets, LL zero-mode counting, and outer-site zero modes in ABC stacks | Anchors the clean ABC LL zero-mode-count tests and the `k=0` outer-site localization tests for ABC trilayer, tetralayer, and pentalayer |
| Koshino, Phys. Rev. B 81, 125304 (2010) | AB as the `N=2` member shared by Bernal and rhombohedral stacking families | Anchors the bilayer Bernal/ABC spectral equivalence checks |

## First-Wave CI Validation

The current fast physics-validation tier is:

1. Bilayer zero-field exact spectrum in the clean BLG subspace.
2. Bilayer zero-field low-energy agreement between the full 4-band model and the package two-band reduction.
3. Bilayer AB/ABC equivalence at zero field and in the LL builder, plus the spinless clean-limit LL zero modes.
4. ABA trilayer mirror-parity decoupling into odd/even sectors at `U=0`, plus explicit recoupling once finite `U` breaks the mirror symmetry.
5. ABA trilayer clean-subspace mirror-sector spectra: the odd block matches a monolayer Dirac Hamiltonian and the even block matches the bilayer-like spectrum with `sqrt(2) * gamma1`.
6. ABA trilayer clean-subspace LL structure: in the mirror basis, the odd block matches the monolayer LL Hamiltonian while the even block matches the bilayer-like LL Hamiltonian with `sqrt(2) * gamma1`.
7. ABA trilayer full-parameter LL mirror structure: with the realistic ABA parameter set and `U=0`, the LL Hamiltonian still block-diagonalizes in the mirror basis, and finite `U` recouples the odd/even sectors.
8. ABA trilayer full-parameter monolayerlike LL block content: in the mirror basis, the odd block exactly matches the monolayerlike block `H_0` from Koshino-McCann once the package parameters are translated as `Delta_eff = Delta - gamma2` and `delta_eff = delta - (gamma2 + gamma5)/2`.
9. ABA trilayer source-aligned bilayerlike LL block content: in the mirror basis, the even block exactly matches the paper's `H_2 = H(sqrt(2)) + W(1/2, 0)` construction once the package is put in the source-aligned convention `Delta=0`, with the `H(sqrt(2))` part realized by a bilayer-like LL Hamiltonian at `gamma1, gamma3, gamma4 -> sqrt(2)` times their Bernal values and the residual `W(1/2, 0)` piece added as a diagonal `gamma2/gamma5` correction on `(A_+, B_+, A_2, B_2)`.
10. ABC trilayer, tetralayer, and pentalayer outer-site zero modes at `k=0` in the clean nearest-neighbor subset.
11. Clean ABC LL zero-mode counting: the LL Hamiltonian has exactly `N` zero modes per valley for rhombohedral `N = 3, 4, 5` in the nearest-neighbor clean subset, with the next LL pinned safely away from zero.
12. ABC trilayer and tetralayer `E ~ k^N` scaling plus low-energy agreement between the full and projected two-band models.
13. Inversion-protected LL valley degeneracy for Bernal bilayer/tetralayer/hexalayer and rhombohedral trilayer/tetralayer/pentalayer, compared through sorted `K/K'` spectra rather than raw LL matrices. The Bernal even-layer checks explicitly use `U=0` and `Delta=0`, while the higher-`N` rhombohedral checks keep `Delta=0` so they stay within the literature-standard parameter surface.
14. ABAB four-layer next-nearest-layer parity in zero field: the `1 -> 3` block carries `diag(gamma2/2, gamma5/2)` while the `2 -> 4` block carries `diag(gamma5/2, gamma2/2)`.

These are implemented in [tests/test_physics_validation.py](/Users/wolft/Dev/contimod_graphene/tests/test_physics_validation.py) and [tests/test_bernal.py](/Users/wolft/Dev/contimod_graphene/tests/test_bernal.py), and rely on the zero-field basis helpers in [src/contimod_graphene/basis.py](/Users/wolft/Dev/contimod_graphene/src/contimod_graphene/basis.py), including the explicit ABA trilayer mirror-basis operator, zero-field unitary, generic layer/block unitaries, and odd/even projectors.

## Deferred / Slower Validation

- Bernal finite-`Delta` coverage currently stops at bilayer zero-field agreement with the package two-band model. Paper-backed finite-`Delta` multilayer and LL validations remain future work.
- Rhombohedral `Delta` is now validated as a distinct even layer-offset term relative to `U`, but only the trilayer meaning is directly paper-backed. The scalar `N>3` extension remains a package convention.
- The rhombohedral `delta` slot is intentionally unused for now. The ABC kernels accept it only for shared-parameter compatibility, and tests pin that it does not affect zero-field, two-band, or LL outputs until a source-backed onsite meaning is adopted.
- The exact rhombohedral fast tier now reaches generic higher-`N` LL structure through pentalayer `K/K'` valley degeneracy and clean-subset LL zero-mode counting.
- The slow rhombohedral regression tier now includes a clean-subset LL field-scaling fit for the lowest positive branch in ABC trilayer and tetralayer, using a moderate field window and explicit `n_cut` stability to check the expected `E ~ B^(N/2)` chirality scaling without touching the fragile low-field warped-LL regime.
- ABA trilayer detailed full-parameter LL-shape regressions remain deferred. The fast suite now covers exact mirror-basis decoupling at realistic ABA parameters, the exact monolayerlike odd-block content, and the exact source-aligned bilayerlike even-block `H_2` construction, but not the approximate bilayerlike low-energy pair ordering, trigonal-warping-driven crossings, or finite-parameter anticrossings yet. The `xi = ±1` to package `K/K'` mapping is now source-backed (`K -> xi = -1`, `K' -> xi = +1`), so the remaining blocker is narrower: once `gamma3` is restored, the bilayerlike `n = -1, 0` pair only survives as adiabatic descendants inside the full even block, and therefore cannot be selected by a fixed sorted-eigenvalue index without adding explicit state-tracking machinery.
- Zero-field valley-degeneracy tests remain deferred because the zero-field public Hamiltonians do not expose a valley switch.
- Bilayer finite-`gamma3` Lifshitz topology, ABA modulo-3 LL anticrossings, and ABC low-field LL triplets belong in a slower regression tier rather than the default fast suite.
