# ML / Geospatial Engine

Owner: Person A (currently unassigned — reassign in docs/roadmap.md if
ownership changes).

## Layout
- `data/` — GeoLife baseline + synthetic Algase-typology anomaly injection
  (gitignored once real/large files land; keep generation scripts in git).
- `features/` — spatial/kinematic feature extraction (tortuosity, entropy,
  distance-to-anchor), Gamma/empirical percentile fitting.
- `training/` — XGBoost + LSTM-Autoencoder training, GroupKFold TOST
  validation pipeline.
- `models/` — serialized artifacts (gitignored — hand off via release
  artifact, not git).

## Contract

All inputs/outputs must conform to `/shared/schema.json`
(`TelemetryPayload` in, `MobilityProfile` + `RiskScoreOutput` out).
