# Physics Validation Plan

This note records the package-specific convention map and the first paper-backed validation wave for `contimod_graphene`. It is intentionally narrower than a literature review: the goal is to explain which claims the test suite asserts today, which paper results those tests come from, and which physics checks are still deferred.

## Package Conventions

- Zero-field full Hamiltonians use the orbital ordering `(A1, B1, A2, B2, ..., AN, BN)`.
- The zero-field public API is valley-fixed. Tests at zero field therefore compare spectra and low-energy structure within that fixed convention rather than asserting explicit `K/K'` degeneracies.
- Landau-level Hamiltonians use asymmetric LL bases that depend on the selected valley. LL tests compare spectra only, not raw matrix elements or basis vectors.
- First-wave analytic checks use explicit clean-subspace parameter overlays instead of package defaults:
  - BLG clean subset: `gamma2=gamma3=gamma4=gamma5=U=Delta=delta=0`
  - ABC clean subset (`N=3,4`): `gamma2=gamma3=gamma4=gamma5=U=Delta=delta=0`
- Tests compare sorted eigenvalues, named-site projector weights, or fitted low-energy exponents. They do not compare raw matrices against paper conventions unless the basis is exposed explicitly.

## Literature Anchors

| Reference | Scope in this repo | Current use |
| --- | --- | --- |
| McCann & Koshino, Rep. Prog. Phys. 76, 056503 (2013) | AB bilayer zero-field 4-band spectrum and low-energy two-band reduction | Anchors the clean BLG exact-band and low-energy reduction tests |
| Koshino & McCann, Phys. Rev. B 80, 165409 (2009) | ABC trilayer and general `N`-layer low-energy reductions | Anchors the ABC `k^3` / `k^4` scaling and full-vs-two-band checks |
| Min & MacDonald, Prog. Theor. Phys. Suppl. 176, 227 (2008) | Nearest-neighbor chiral doublets and outer-site zero modes in ABC stacks | Anchors the `k=0` outer-site localization tests for ABC trilayer and tetralayer |
| Koshino, Phys. Rev. B 81, 125304 (2010) | AB as the `N=2` member shared by Bernal and rhombohedral stacking families | Anchors the bilayer Bernal/ABC spectral equivalence checks |

## First-Wave CI Validation

The current fast physics-validation tier is:

1. Bilayer zero-field exact spectrum in the clean BLG subspace.
2. Bilayer zero-field low-energy agreement between the full 4-band model and the package two-band reduction.
3. Bilayer AB/ABC equivalence at zero field and in the LL builder, plus the spinless clean-limit LL zero modes.
4. ABC trilayer and tetralayer outer-site zero modes at `k=0`.
5. ABC trilayer and tetralayer `E ~ k^N` scaling plus low-energy agreement between the full and projected two-band models.

These are implemented in [tests/test_physics_validation.py](/Users/wolft/Dev/contimod_graphene/tests/test_physics_validation.py) and rely on the zero-field basis helpers in [src/contimod_graphene/basis.py](/Users/wolft/Dev/contimod_graphene/src/contimod_graphene/basis.py).

## Deferred / Slower Validation

- ABA trilayer mirror-parity block tests remain deferred until the package exposes an explicit mirror/parity basis transform or projector helper.
- Zero-field valley-degeneracy tests remain deferred because the zero-field public Hamiltonians do not expose a valley switch.
- Bilayer finite-`gamma3` Lifshitz topology, ABA modulo-3 LL anticrossings, and ABC low-field LL triplets belong in a slower regression tier rather than the default fast suite.
- The Bernal full zero-field kernel still ignores `Delta` even though the bilayer two-band reduction uses it. That is a model-surface inconsistency worth resolving before adding paper-backed finite-asymmetry tests.
