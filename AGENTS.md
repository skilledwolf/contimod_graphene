# contimod_graphene Agent Guide

## Mission
- `contimod_graphene` is a standalone multilayer-graphene model package in this repo family.
- The standing goal is to make it a clean, trustworthy, reusable package for Bernal and rhombohedral graphene Hamiltonians, parameter sets, basis metadata, and related single-particle helpers.
- Prefer a crisp boundary: `contimod_graphene` should own low-level graphene modeling primitives, while `contimod` should own discretization, mesh/state containers, and many-body workflows unless there is a strong reason to move a reusable piece down here.

## Scope
- Core package code lives in `src/contimod_graphene/`.
- Tests live in `tests/`.
- User-facing documentation and usage notes live in `README.md`, `docs/`, and `examples/`.
- Keep the package as self-contained as practical. Do not introduce `contimod` dependencies into the core library without an explicit architecture decision.

## Environment
- Python `>=3.10`.
- Dev install: `python -m pip install -e .` or `hatch env create && hatch shell`.
- Quick import check: `python -c "import contimod_graphene; print(contimod_graphene.__version__)"` when version metadata is available.
- Tests: `pytest -ra`.

## Working Rules
- Start by reading `Roadmap.md` and `git status --short --branch`.
- Treat the worktree as potentially dirty. Never overwrite or revert unrelated user changes.
- If you discover a bug, cleanup item, packaging issue, API mismatch, missing validation, or doc gap, add it to `Roadmap.md` immediately instead of leaving it in scratch notes.
- If you work on the standalone public API or parameter-management surface, read `PUBLIC_API_PLAN.md` first and keep it aligned with any implementation decisions you make.
- When you start a task, update its status in `Roadmap.md`.
- When you finish, block, split, supersede, add, or remove task entries as appropriate in the same unit of work.
- Keep task notes concrete: affected paths, expected validation, and follow-up work that remains.

## Validation
- Use layered validation, not just smoke imports.
- For touched code, add or update focused tests where feasible.
- For numerical kernels, prefer deterministic checks beyond shape/hermiticity alone when practical.
- If docs or examples claim a workflow is supported, make sure there is a matching validation story.

## Commits
- Create commits after meaningful units of work.
- Stage only the files relevant to that unit.
- Update `Roadmap.md` in the same commit when task state changes or new tasks are discovered.
- Prefer conventional commit messages such as `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, or `chore:`.

## Near-Term Direction
- Strengthen the package as a standalone graphene-model library that can be used directly, with `contimod` treated as one downstream consumer rather than the package's reason for existence.
- Use the concrete API and parameter-management direction in `PUBLIC_API_PLAN.md` as the default starting point unless a better replacement is written down in the repo.
- Keep docs/examples usable without assuming `contimod`, except where a document is explicitly about downstream integration.
- Be conservative about adding optional dependencies: they should unlock a clear surface such as BZ/path helpers, plotting helpers, or symmetry classification.
