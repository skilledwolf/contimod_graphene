from __future__ import annotations

import importlib.util
import pathlib


DOCS_ROOT = pathlib.Path(__file__).resolve().parent
STATIC_DIR = DOCS_ROOT / "_static"
EXAMPLES_SCRIPT = DOCS_ROOT.parent / "examples" / "standalone_gallery.py"


def _load_examples_module():
    spec = importlib.util.spec_from_file_location(
        "contimod_graphene_standalone_gallery",
        EXAMPLES_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXAMPLES = _load_examples_module()


def _ensure_static_dir() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def generate_abc_trilayer_bandstructure(
    *,
    n_k: int = 400,
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    outdir = pathlib.Path(output_dir) if output_dir is not None else STATIC_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    result = EXAMPLES.abc_trilayer_bandstructure(outdir=outdir, n_k=n_k)
    return pathlib.Path(result["output_path"])


def generate_blg_landau_level_fan(
    *,
    n_b: int = 80,
    n_cut: int = 40,
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    outdir = pathlib.Path(output_dir) if output_dir is not None else STATIC_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    result = EXAMPLES.blg_landau_level_fan(outdir=outdir, n_b=n_b, n_cut=n_cut)
    return pathlib.Path(result["output_path"])


def generate_aba_trilayer_bandstructure(
    *,
    n_k: int = 400,
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    outdir = pathlib.Path(output_dir) if output_dir is not None else STATIC_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    result = EXAMPLES.aba_trilayer_bandstructure(outdir=outdir, n_k=n_k)
    return pathlib.Path(result["output_path"])


def main(
    *,
    output_dir: pathlib.Path | str | None = None,
    n_k: int = 400,
    n_b: int = 80,
    n_cut: int = 40,
) -> dict[str, object]:
    outdir = pathlib.Path(output_dir) if output_dir is not None else STATIC_DIR
    _ensure_static_dir() if output_dir is None else outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "abc_trilayer": str(generate_abc_trilayer_bandstructure(n_k=n_k, output_dir=outdir)),
        "blg_landau_levels": str(generate_blg_landau_level_fan(n_b=n_b, n_cut=n_cut, output_dir=outdir)),
        "aba_trilayer": str(generate_aba_trilayer_bandstructure(n_k=n_k, output_dir=outdir)),
    }
    outputs["output_paths"] = list(outputs.values())
    return outputs


if __name__ == "__main__":
    main()
