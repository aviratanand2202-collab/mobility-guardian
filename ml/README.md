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

## Contract & Inference Architecture

All inputs/outputs conform to `/shared/schema.json` (`TelemetryPayload` in, `RiskScoreOutput` out) with an embedded deterministic decision layer (`DecisionOutput`).

Comprehensive deployment contracts and architectural specifications are documented in [`docs/inference.md`](file:///c:/Projects/mobility-guardian/docs/inference.md):
- **New User Flow**: `current movement -> COLD_START / population baseline -> safe-area context -> history accumulation -> personal profile -> personalized inference`
- **Existing User Flow**: `historical trajectory folder/profile + current trajectory/telemetry -> personalized inference`
- **Locked Horizons**: `120s` (2m Micro), `360s` (6m Operational Short), `600s` (10m Medium), `840s` (14m Operational Extended)
- **Geospatial Context**: Safe areas (circle, polygon, bbox) and algorithmic spatial familiarity (DBSCAN anchors)
- **Deterministic Decision Output**: 6 canonical contextual states (`SAFE / NORMAL`, `OUTSIDE_SAFE_AREA`, `FAMILIAR_MOVEMENT`, `UNFAMILIAR_MOVEMENT`, `SUSPICIOUS`, `CRITICAL`) with strict behavioral dominance and non-clinical disclaimers.

## GeoLife Trajectory Preprocessing (Chunk 1)

### Dataset Placement
- **Raw Data Location**: `ml/data/raw/geolife/<user_id>/Trajectory/*.plt`
- User folders (`000/`, `001/`, ..., `181/`) are discovered dynamically without hard-coding IDs, counts, or paths. Raw files are strictly read-only and never modified.

### Running Preprocessing
From project root or `ml/`:
```bash
# Run on complete dataset
py ml/src/data.py

# Optional parameters:
py ml/src/data.py --raw-dir ml/data/raw/geolife --output-dir ml/data/processed --num-workers 8 --max-users 10
```

### Running Tests
```bash
py -m pytest ml/tests/test_data.py -v
```

### Processed Output
- **Canonical Trajectory Data**: `ml/data/processed/trajectories.parquet` (compressed with Snappy, canonical schema: `user_id`, `trajectory_id`, `timestamp`, `latitude`, `longitude`, `altitude`, `raw_days`, `date_str`, `time_str`, `subsecond_seq`, `timestamp_collision`).
- **Data Quality Report**:
  - `ml/data/processed/data_quality_report.json`
  - `ml/data/processed/data_quality_report.md`
- **Exploratory Analysis Notebook**: `ml/notebooks/01_data_exploration.ipynb`

### Cleaning & Validation Performed
- **Physical Bounds**: Validates latitude $\in [-90, 90]$ and longitude $\in [-180, 180]$.
- **Timestamp Integrity**: Validates datetime parsing; removes uninitialized RTC hardware clock resets (e.g. pre-2005 default epoch `2000-01-01`).
- **Exact Deduplication**: Removes redundant rows matching across all fields.
- **Identical-Coordinate Collision Removal**: Deduplicates same-second logs where coordinates are identical (`\Delta d = 0`).
- **Same-Second Spatial Preservation**: Legitimate distinct spatial observations sharing the same second are preserved and annotated with `timestamp_collision = True` and original source order `subsecond_seq` (0, 1, ...).
- **Chronological Ordering**: Validates chronological ordering within each trajectory; applies stable sorting when out-of-order points are detected.
- **Missing Values**: Removes records with missing coordinates or timestamps.
- **Sampling Interval Statistics**: Derived directly from actual timestamps for $dt > 0$, protecting downstream calculations from division by zero.

### What Is Intentionally NOT Removed (Conservative Cleaning)
- **Same-Second Observations**: Distinct spatial observations within identical seconds are preserved with explicit collision metadata.
- **Stationary Points**: Successive records with identical coordinates across advancing timestamps are preserved (essential for dwell/anchor point detection).
- **Kinematic Anomalies / High Speeds**: Fast movements or apparent anomalies are preserved (essential for downstream wandering/anomaly detection).
- **Altitude Values**: Preserved as-is, including -777 (GeoLife sensor uncalibrated flag).
- **No Early Resampling / Segmentation**: No trip segmentation, coordinate interpolation, or windowing at this stage.

### Transportation Context & Downstream Kinematics
GeoLife records real-world multimodal mobility, containing diverse transportation contexts ranging from pedestrian walking to high-speed rail and commercial aircraft flights (e.g. trans-Pacific flights crossing northern latitudes and the 180° Anti-Meridian). All valid high-speed transportation trajectories are preserved in the canonical dataset without alteration. Downstream kinematic and behavioral modeling stages must distinguish valid transit modes from pedestrian wandering or geofence boundary breach anomalies.
