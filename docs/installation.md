# Installation

For most users, install directly from GitHub:

```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```

Quick smoke check:

```bash
python -c "import contimod_graphene as cg; print(cg.list_parameter_sets())"
```

If you are running examples or tests on Apple Silicon, prefer CPU for now:

```bash
JAX_PLATFORMS=cpu python examples/standalone_quickstart.py
```

The Metal backend currently hits known JAX failures in this repo.

## Develop Locally

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/skilledwolf/contimod_graphene.git
cd contimod_graphene
pip install -e ".[dev]"
```

If you prefer `hatch`:

```bash
hatch env create
hatch shell
```

## Containerized Jupyter

If you want a throwaway Jupyter environment, `repo2docker` works:

```bash
jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```

Requirements: Docker plus `repo2docker`. This path is convenient, but not tested on every platform.
