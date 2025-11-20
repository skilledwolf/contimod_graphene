"""Parameter management for graphene models (JSON-backed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_DEFAULT_JSON = (_HERE.parent.parent / "data" / "params.json").resolve()

with _DEFAULT_JSON.open() as f:
    _DATA = json.load(f)

_ALIASES = {**_DATA.get("aliases", {})}

graphene_params = _DATA["sets"]["slg"]
graphene_params_BLG = _DATA["sets"]["blg"]
graphene_params_TLG = _DATA["sets"]["tlg"]
graphene_params_4LG = _DATA["sets"]["4lg"]


def _resolve_kind(kind: str) -> str:
    key = kind.lower()
    if key in _DATA["sets"]:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"Unknown graphene parameter set '{kind}'")


def get_params(kind: str | Dict[str, Any]) -> dict:
    """Return params by name (or pass-through dict). Accepts JSON path too."""
    if isinstance(kind, dict):
        return dict(kind)
    if Path(str(kind)).exists():
        return load(str(kind))
    key = _resolve_kind(str(kind))
    return dict(_DATA["sets"][key])


def load(path: str) -> dict:
    """Load a parameter dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def list_sets():
    return list(_DATA["sets"].keys())
