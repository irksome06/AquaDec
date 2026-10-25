# Contributing

## Branches

- Branch from `main`.
- Name branches by intent, for example `feat/svm-tuning` or `fix/schema-ranges`.
- Keep one model, config, or serving change per pull request when you can.

## Frozen split

`data/splits/splits.csv` is the shared train/test location assignment. Do not regenerate it in ordinary PRs. If the split must change, say why in the PR, update `configs/config.yaml` checksums if `clean_data.csv` also changed, and re-run evaluation for **all four** models.

## Pull requests

Use the checklist in `.github/pull_request_template.md`. CI is expected to run `pytest` from this directory (`lake-bloom-severity/`).

Do not commit files under `outputs/`. Small files under `results/` and serving artifacts under `artifacts/<model>/` belong in git after a training run.

## Serving contract

Training features must match `serving/input_schema.json`. Do not add `year`, `distance_to_water_m`, `region`, or `data_provider` to the model input.
