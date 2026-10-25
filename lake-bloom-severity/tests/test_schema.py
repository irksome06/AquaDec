import pytest

from model.paths import KNOWN_MODELS, load_config
from model.registry import build_pipeline
from serving.schema import SchemaError, load_schema, validate_payload


FORBIDDEN = {"year", "distance_to_water_m", "region", "data_provider"}


def test_feature_list_excludes_forbidden_columns():
    cfg = load_config()
    features = set(cfg["data"]["feature_cols"])
    assert features.isdisjoint(FORBIDDEN)
    assert FORBIDDEN.issubset(set(cfg["data"]["forbidden_cols"]))
    assert "month" in features
    assert "tmax_c" in features
    assert "NDCI" in features


def test_each_registry_pipeline_builds():
    for name in KNOWN_MODELS:
        pipe = build_pipeline(name)
        assert [step[0] for step in pipe.steps][0] == "preprocessor"
        assert pipe.named_steps["classifier"] is not None


def test_schema_rejects_forbidden_and_missing_fields():
    schema = load_schema()
    required = {name: 1 for name in schema["required"]}
    with pytest.raises(SchemaError, match="forbidden"):
        validate_payload({**required, "region": "midwest"}, schema)
    incomplete = dict(required)
    incomplete.pop("tmax_c")
    with pytest.raises(SchemaError, match="missing"):
        validate_payload(incomplete, schema)


def test_schema_accepts_complete_numeric_row():
    schema = load_schema()
    row = {}
    for field in schema["fields"]:
        if field["name"] == "month":
            row["month"] = 7
        else:
            row[field["name"]] = (field["min"] + field["max"]) / 2
    frame = validate_payload(row, schema)
    assert list(frame.columns) == schema["required"]
    assert len(frame) == 1
