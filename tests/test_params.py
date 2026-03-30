from __future__ import annotations

import json

import pytest

from contimod_graphene import GrapheneTBParameters, list_parameter_sets, load_parameter_set
from contimod_graphene.params import graphene_params_BLG, graphene_params_TLG


def test_preset_alias_resolution_and_mapping_compatibility():
    params = GrapheneTBParameters.preset("bilayer")

    assert params.preset_name == "blg"
    assert dict(params)["gamma0"] == graphene_params_BLG["gamma0"]
    assert params["Delta"] == graphene_params_BLG["Delta"]


def test_replace_updates_known_keys_and_tracks_explicit_extras():
    params = GrapheneTBParameters.preset("tlg").replace(U=12.5, lambda1_eff=1.2)

    assert params["U"] == pytest.approx(12.5)
    assert params["lambda1_eff"] == pytest.approx(1.2)
    assert params.extras["lambda1_eff"] == pytest.approx(1.2)


def test_load_parameter_set_accepts_name_path_and_object(tmp_path):
    params = load_parameter_set("tlg")
    path = tmp_path / "custom_params.json"
    params.replace(U=7.0, gamma5=19.0).to_json(path)

    loaded_from_path = load_parameter_set(path)
    loaded_from_object = load_parameter_set(params)

    assert loaded_from_path["U"] == pytest.approx(7.0)
    assert loaded_from_path["gamma5"] == pytest.approx(19.0)
    assert loaded_from_object is params


def test_round_trip_json_preserves_payload(tmp_path):
    params = graphene_params_TLG.replace(U=18.0, lambda2_eff=0.25)
    path = tmp_path / "roundtrip.json"

    params.to_json(path)
    restored = GrapheneTBParameters.from_json(path)

    assert restored.to_dict() == params.to_dict()
    assert json.loads(path.read_text()) == params.to_dict()


def test_load_parameter_set_requires_complete_core_mapping():
    with pytest.raises(ValueError, match="Missing required graphene parameters"):
        load_parameter_set({"gamma0": 1.0})


def test_validate_for_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown graphene family"):
        graphene_params_TLG.validate_for("twisted")


def test_list_parameter_sets_returns_canonical_names():
    assert list_parameter_sets() == ["slg", "blg", "tlg", "4lg"]
