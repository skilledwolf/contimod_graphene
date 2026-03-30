# Examples

This directory contains two kinds of example material:

- Maintained standalone examples that exercise the package-native public API.
- Companion notebooks that provide longer exploratory workflows.

## Maintained Standalone Entry Point

- `standalone_quickstart.py`
  - Lightweight public-API example using `GrapheneTBParameters`, `BernalMultilayer`, and `RhombohedralMultilayer`.
  - Writes a small JSON summary under `examples/outputs/` by default.
  - Intended to stay CPU-smoke-testable.
- `standalone_gallery.py`
  - Figure-producing standalone examples for ABC band structures, bilayer Landau-level fans, and ABA band scans.
  - The maintained ABC band-structure path pins `Delta=0.0` so the example isolates the effect of `U`.
  - Matches the maintained snippets shown in `docs/examples.md`.

## Companion Notebooks

- `bandstructure_plots.ipynb`
  - Rhombohedral band-structure exploration and effective two-band comparisons.
- `bernal_bands_LL.ipynb`
  - Bernal stacking band-structure and Landau-level exploration.
- `landau_level_fans.ipynb`
  - Landau-level fan diagrams for multilayer graphene models.

These notebooks are useful longer-form references, but the standalone scripts
above are the maintained starting point for the public API.

## Downstream Integration

- `contimod_example.ipynb`
  - Explicit downstream integration example.
  - Requires the external `contimod` package and is not part of the standalone
    default usage story.
