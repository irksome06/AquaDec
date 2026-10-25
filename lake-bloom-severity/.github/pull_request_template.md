## Summary

<!-- What changed and why. -->

## Checklist

- [ ] Did not train on `year`, `distance_to_water_m`, `region`, or `data_provider`
- [ ] Did not rewrite `data/splits/splits.csv` unless this PR is explicitly about the split
- [ ] `configs/config.yaml` checksum matches `data/processed/clean_data.csv` if data changed
- [ ] Serving schema (`serving/input_schema.json`) still matches training features
- [ ] Tests pass (`pytest` from `lake-bloom-severity/`)
- [ ] If a model was retrained: updated `artifacts/<model>/`, `results/<model>/`, and comparison outputs
