# Installation

## Option 1: User Installation (Preferred)

If you just want to use the package without modifying it, you can install it directly from GitHub:

```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```

This will allow you to import the package in Python:
```python
import contimod_graphene
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
    pip install -e .
    ```

   Or using `hatch` (recommended for managing environments):
   ```bash
   hatch env create
   hatch shell
   ```

## Option 3: Docker (Cross-platform)

To spin up a containerized Jupyter environment:

```bash
jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```
