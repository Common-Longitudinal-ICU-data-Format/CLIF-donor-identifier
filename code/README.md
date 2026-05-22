# `code/`

Entry point for the CLIF donor identification analysis.

## Setup

```bash
uv sync                                          # install dependencies
cp ../config/config_template.json ../config/config.json
# edit config/config.json with your site_name, tables_path, file_type, timezone
```

## Run

```bash
uv run python code/01_potential_donor_identifier.py
```

Reads the configured CLIF tables (`patient`, `hospitalization`, `adt`, `vitals`, `labs`, `respiratory_support`, `crrt_therapy`, `hospital_diagnosis`, `microbiology_culture`, `patient_assessments`), applies the CALC and CLIF donor filters within the 2020–2025 window, and writes all outputs to `output/final/`.

stdout is mirrored to `output/final/run_log.txt` for the most recent run.

See the [project README](../README.md) for cohort definitions, output specification, and required CLIF table columns.
