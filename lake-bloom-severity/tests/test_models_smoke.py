import json
from pathlib import Path

import pytest

from model.paths import KNOWN_MODELS, project_root
from serving.predictor import PredictError, predict
from serving.schema import load_schema

ARTIFACTS = project_root() / "artifacts"


def _sample_row():
    schema = load_schema()
    row = {}
    for field in schema["fields"]:
        if field["name"] == "month":
            row["month"] = 8
        else:
            row[field["name"]] = (field["min"] + field["max"]) / 2
    return row


@pytest.mark.parametrize("model_name", KNOWN_MODELS)
def test_predict_smoke_if_artifact_present(model_name):
    path = ARTIFACTS / model_name / "model.joblib"
    if not path.exists():
        pytest.skip(f"no artifact for {model_name}")
    result = predict(model_name, _sample_row())
    assert result["predicted_severity"] in {1, 2, 3, 4}
    assert set(result["probabilities"]) == {"1", "2", "3", "4"}
    card = json.loads((ARTIFACTS / model_name / "model_card.json").read_text(encoding="utf-8"))
    assert "year" not in card["features"]
    assert "region" not in card["features"]


def test_unknown_model_raises():
    with pytest.raises(PredictError):
        predict("decision_tree", _sample_row())
