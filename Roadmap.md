# Roadmap

This file is the shared backlog for `contimod_graphene` cleanup, packaging, validation, and API-shaping work. Add tasks here as soon as they are discovered. Update task status in the same unit of work when tasks start, finish, split, get blocked, or are no longer relevant.

## Clean Worktree Plan
- [x] Add repo-root `AGENTS.md` and `Roadmap.md` files so package-specific guidance and backlog tracking live in `contimod_graphene`, not only in downstream repos.

## Active Priorities
- [ ] Decide whether to promote `contimod_graphene` from a `contimod`-branded helper into a clearly standalone multilayer-graphene package. If yes, define the ownership boundary explicitly: low-level Bernal/ABC kernels, parameter sets, basis/symmetry helpers, and lightweight single-particle utilities should live here, while `contimod` should keep discretization, mesh/state containers, and many-body workflows (SCF, susceptibility, TDHF, superconductivity).
- [ ] Decide whether the package should keep the current distribution/import naming or adopt a more independent name. If a rename is chosen, define the migration strategy separately for the distribution name and the Python import path rather than changing both casually.
- [ ] Decide whether to add a higher-level model surface on top of the current kernel-first API, for example lightweight model objects or helpers that package parameters, valley/spin lifting, basis metadata, and batched evaluation without depending on `contimod`.
- [ ] Audit docs and examples so the default usage story stands on its own without assuming `contimod`, except for explicitly downstream-integration documents.
- [ ] Evaluate `ase` as an optional dependency for Brillouin-zone/path/plotting helpers if that produces a cleaner standalone single-particle API.
- [ ] Evaluate `spglib` as an optional dependency only if symmetry classification becomes part of the intended public surface.
- [ ] Strengthen validation beyond shape/hermiticity smoke tests for the core Hamiltonian builders, especially where the package intends to make standalone scientific claims.

## Blocked / Needs Decision
- [ ] Decide how much of the existing `contimod` wrapper surface, if any, should move into `contimod_graphene` versus remaining as downstream integration code.

## Seed Notes
- The source tree is already mostly self-contained and does not depend on `contimod` in core modules, which makes a stronger standalone positioning plausible.
- The current public surface is still shaped like a helper package in its naming, docs framing, and kernel-centric API.
