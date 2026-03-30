# Roadmap

This file is the shared backlog for `contimod_graphene` cleanup, packaging, validation, and API-shaping work. Add tasks here as soon as they are discovered. Update task status in the same unit of work when tasks start, finish, split, get blocked, or are no longer relevant.

## Clean Worktree Plan
- [x] Add repo-root `AGENTS.md` and `Roadmap.md` files so package-specific guidance and backlog tracking live in `contimod_graphene`, not only in downstream repos.
- [x] Reframe package-facing guidance and metadata so `contimod_graphene` is presented as a standalone package in its own right rather than as a helper package, while still documenting `contimod` as an important downstream integration.

## Active Priorities
- [ ] Continue sharpening the standalone-package boundary: low-level Bernal/ABC kernels, parameter sets, basis/symmetry helpers, and lightweight single-particle utilities should live here, while `contimod` should keep discretization, mesh/state containers, and many-body workflows (SCF, susceptibility, TDHF, superconductivity) unless a reusable primitive clearly belongs downstream.
- [ ] Decide whether the package should keep the current distribution/import naming or adopt a more independent name. If a rename is chosen, define the migration strategy separately for the distribution name and the Python import path rather than changing both casually.
- [ ] Decide whether to add a higher-level model surface on top of the current kernel-first API, for example lightweight model objects or helpers that package parameters, valley/spin lifting, basis metadata, and batched evaluation without depending on `contimod`.
- [ ] Design and land an explicit standalone public API layer instead of exposing mostly module namespaces. Candidate direction: keep `bernal.py` / `rhombohedral.py` as low-level kernels, add a small `api.py` or `models/` surface with stable user-facing constructors, and make `src/contimod_graphene/__init__.py` export that surface intentionally. Validation: import-level API smoke tests plus docs/README examples updated to use the new entry points.
- [ ] Replace the raw-dict parameter surface with a more deliberate parameter-management layer in `src/contimod_graphene/params.py`: preserve JSON-backed presets, but add validated overlay/update helpers and a stable parameter object or mapping contract so users can inspect, copy, override, and serialize parameters without guessing which keys matter. Validation: focused tests for preset loading, alias resolution, overlay semantics, and round-tripping custom parameter files.
- [ ] Remove duplicated low-level helpers and make module ownership obvious. Right now `src/contimod_graphene/utils.py` and `src/contimod_graphene/basis.py` both define layer/sublattice coordinate helpers; decide which module owns geometry/basis metadata versus numerical helper functions and collapse the duplicates behind one documented import path. Validation: update imports, remove dead helpers, and keep existing basis/LL tests green.
- [ ] Fix the current model-identity mismatch in `src/contimod_graphene/rhombohedral.py`: the rhombohedral builders still default to `graphene_params_BLG`, and some wrapper docstrings still mention Bernal at the bottom of the file. Align defaults/docs with the intended ABC model presets before treating the standalone API as settled. Validation: targeted tests for default-preset selection plus a docs/examples audit of the rhombohedral entry points.
- [ ] Audit docs and examples so the default usage story stands on its own without assuming `contimod`, except for explicitly downstream-integration documents.
- [ ] Evaluate `ase` as an optional dependency for Brillouin-zone/path/plotting helpers if that produces a cleaner standalone single-particle API.
- [ ] Evaluate `spglib` as an optional dependency only if symmetry classification becomes part of the intended public surface.
- [ ] Strengthen validation beyond shape/hermiticity smoke tests for the core Hamiltonian builders, especially where the package intends to make standalone scientific claims.

## Blocked / Needs Decision
- [ ] Decide how much of the existing `contimod` wrapper surface, if any, should move into `contimod_graphene` versus remaining as downstream integration code.

## Seed Notes
- The source tree is already mostly self-contained and does not depend on `contimod` in core modules, which supports the standalone package positioning.
- The remaining follow-on work is less about whether this is a real package and more about how polished and complete that standalone surface should become.
