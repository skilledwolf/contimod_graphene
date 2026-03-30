from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np

import contimod_graphene as cg


matplotlib.use("Agg")


def _load_example_module(relpath: str, module_name: str):
    example_path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(module_name, example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _notebook_code(relpath: str) -> str:
    notebook_path = Path(__file__).resolve().parents[1] / relpath
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            parts.append("".join(cell.get("source", [])))
    return "\n\n".join(parts)


def test_readme_and_usage_quickstart_surface_smoke(tmp_path):
    params = cg.GrapheneTBParameters.preset("tlg").replace(U=20.0)
    model = cg.RhombohedralMultilayer(n_layers=3, params=params)

    h0 = np.asarray(model.hamiltonian(0.1, 0.0))
    h2 = np.asarray(model.two_band_hamiltonian(0.1, 0.0))
    hs = np.asarray(model.hamiltonian_batch([[0.0, 0.0], [0.1, 0.0]], jit=False))
    h_ll = np.asarray(
        cg.BernalMultilayer(n_layers=2).landau_level_hamiltonian(
            10.0,
            n_cut=6,
            valley="K",
        )
    )

    assert h0.shape == (6, 6)
    assert h2.shape == (2, 2)
    assert hs.shape == (2, 6, 6)
    assert h_ll.shape == (22, 22)

    params_path = tmp_path / "params.json"
    params.to_json(params_path)
    restored = cg.GrapheneTBParameters.from_json(params_path)

    assert restored.to_dict() == params.to_dict()
    assert cg.list_parameter_sets() == ["slg", "blg", "tlg", "4lg"]


def test_standalone_quickstart_example_writes_summary(tmp_path):
    module = _load_example_module(
        "examples/standalone_quickstart.py",
        "standalone_quickstart_example",
    )

    result = module.main(outdir=tmp_path, num_k=17, ll_n_cut=4)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    saved = json.loads(summary_path.read_text())
    assert saved["parameter_preset"] == "tlg"
    assert saved["available_presets"] == ["slg", "blg", "tlg", "4lg"]
    assert saved["zero_field_shape"] == [6, 6]
    assert saved["two_band_shape"] == [2, 2]
    assert saved["band_shape"] == [17, 6]
    assert saved["landau_level_shape"] == [14, 14]
    assert np.isfinite(saved["band_extrema_mev"]).all()


def test_standalone_gallery_example_writes_expected_artifacts(tmp_path):
    module = _load_example_module(
        "examples/standalone_gallery.py",
        "standalone_gallery_example",
    )

    result = module.main(outdir=tmp_path, n_k=31, n_b=6, n_cut=8)

    output_paths = [Path(path) for path in result["output_paths"]]
    assert len(output_paths) == 3
    for path in output_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    assert result["abc_trilayer"]["bands_full_shape"] == (31, 6)
    assert result["abc_trilayer"]["bands_low_shape"] == (31, 2)
    assert result["aba_trilayer"]["bands_shape"] == (31, 6)
    assert result["blg_landau_levels"]["n_fields"] == 6
    assert result["blg_landau_levels"]["levels_per_field"] > 0


def test_docs_generate_example_figures_smoke(tmp_path):
    module = _load_example_module(
        "docs/generate_example_figures.py",
        "docs_generate_example_figures",
    )

    result = module.main(output_dir=tmp_path, n_k=21, n_b=4, n_cut=6)

    output_paths = [Path(path) for path in result["output_paths"]]
    assert len(output_paths) == 3
    for path in output_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_standalone_notebooks_use_public_api_without_contimod():
    for relpath in (
        "examples/bandstructure_plots.ipynb",
        "examples/landau_level_fans.ipynb",
        "examples/bernal_bands_LL.ipynb",
    ):
        source = _notebook_code(relpath)
        assert "import contimod as cm" not in source
        assert "GrapheneTBParameters" in source
        assert "BernalMultilayer" in source or "RhombohedralMultilayer" in source


def test_contimod_example_notebook_stays_explicitly_downstream():
    source = _notebook_code("examples/contimod_example.ipynb")
    assert "import contimod as cm" in source
