# Installation

For most users:

```bash
pip install contimod-graphene
```

Quick smoke check:

```bash
python -c "import contimod_graphene as cg; print(cg.list_parameter_sets())"
```

The distribution name on PyPI uses a hyphen, but the Python import remains `contimod_graphene`.

If you want the unreleased `main` branch instead of the latest published version:

```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```

If you are running examples or tests on Apple Silicon, prefer CPU for now:

```bash
JAX_PLATFORMS=cpu python examples/standalone_quickstart.py
```

The Metal backend currently hits known JAX failures in this repo.

## Develop Locally

Clone the repository and run `pip install -e ".[dev]"`.

If you prefer `hatch`, use `hatch env create` and `hatch shell`.
