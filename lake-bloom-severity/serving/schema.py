"""Validate prediction payloads against serving/input_schema.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "serving" / "input_schema.json"


class SchemaError(ValueError):
    pass


def load_schema(path: Path | None = None) -> dict:
    schema_path = path or SCHEMA_PATH
    if not schema_path.exists():
        raise SchemaError(
            f"Missing {schema_path}. Train a model first so input_schema.json is generated."
        )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _as_records(payload) -> list[dict]:
    if isinstance(payload, dict):
        if "instances" in payload:
            rows = payload["instances"]
            if not isinstance(rows, list) or not rows:
                raise SchemaError("`instances` must be a non-empty list of objects")
            return rows
        return [payload]
    if isinstance(payload, list):
        if not payload:
            raise SchemaError("payload list is empty")
        if not all(isinstance(row, dict) for row in payload):
            raise SchemaError("list payload must contain objects")
        return payload
    raise SchemaError("payload must be an object, a list of objects, or {instances: [...]}")


def validate_payload(payload, schema: dict | None = None) -> pd.DataFrame:
    schema = schema or load_schema()
    required = list(schema["required"])
    forbidden = set(schema.get("forbidden") or [])
    field_meta = {item["name"]: item for item in schema.get("fields", [])}

    records = _as_records(payload)
    rows = []
    for i, record in enumerate(records):
        extra_forbidden = sorted(set(record) & forbidden)
        if extra_forbidden:
            raise SchemaError(
                f"row {i}: forbidden fields are not used at inference: {extra_forbidden}"
            )
        missing = [name for name in required if name not in record]
        if missing:
            raise SchemaError(f"row {i}: missing required fields: {missing}")
        cleaned = {}
        for name in required:
            value = record[name]
            meta = field_meta.get(name, {})
            try:
                number = int(value) if meta.get("type") == "integer" else float(value)
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"row {i}: field `{name}` must be numeric, got {value!r}") from exc
            low = meta.get("min")
            high = meta.get("max")
            # Allow a small margin outside the training min/max.
            if low is not None and number < float(low) - abs(float(low)) * 0.25 - 1e-6:
                raise SchemaError(f"row {i}: `{name}`={number} is below expected range")
            if high is not None and number > float(high) + abs(float(high)) * 0.25 + 1e-6:
                raise SchemaError(f"row {i}: `{name}`={number} is above expected range")
            if name == "month" and not 1 <= int(number) <= 12:
                raise SchemaError(f"row {i}: month must be 1-12")
            cleaned[name] = int(number) if name == "month" else number
        rows.append(cleaned)
    return pd.DataFrame(rows, columns=required)
