# Installation

## Option 1: GitHub Install

If you just want to use the package without modifying it, install it directly from GitHub:

```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```

This gives you the standalone package import in Python:
```python
import contimod_graphene as cg

print(cg.list_parameter_sets())
```

## Option 2: Developer Installation

If you plan to contribute to the package or modify the source code:

1.  Clone the repository:
    ```bash
    git clone https://github.com/skilledwolf/contimod_graphene.git
    cd contimod_graphene
    ```

2.  Install in editable mode:
    ```bash
    pip install -e ".[dev]"
    ```

   Or, if you use `hatch`, create and enter the managed environment:
   ```bash
   hatch env create
   hatch shell
   ```

3.  Run a quick smoke check:
    ```bash
    python -c "import contimod_graphene as cg; print(cg.list_parameter_sets())"
    ```

## Option 3: Docker (Cross-platform)

To spin up a containerized Jupyter environment:

```bash
jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```
