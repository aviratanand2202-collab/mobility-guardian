# System Limitations & Future Work

All metrics reported in the evaluation section are evaluated on the
synthetically augmented mobility dataset. Real-world performance on
clinical populations will likely exhibit higher variance and requires
future clinical validation.

- **Synthetic-to-Real Domain Gap** — The system is evaluated on
  synthetically injected anomalies based on the Algase Wandering Typology.
  Real-world cognitive impairment trajectories are highly heterogeneous.
  Future work requires clinical validation against annotated real-world
  wandering events.

- **Low-N Percentile Estimator Maturity** — The Gamma distribution
  parametric fit bounds early predictions, but the system is inherently
  fragile during the first ~30 trips before empirical quantiles stabilize.

- **Autocorrelation Constraints** — The k-window trigger logic enforces
  strict non-overlapping temporal windows to prevent autocorrelation from
  causing cascading false positives, which places a hard physical floor
  on early-warning speed.

- **PDR Drift in Extended Signal Loss** — The Pedestrian Dead Reckoning
  fallback relies on uncorrected smartphone IMU fusion. Extended indoor
  pacing (approaching the 15-minute GPS blind-spot limit) risks
  accumulating sufficient drift to falsely exceed the 50m Tier 2
  threshold.

- **Tier 2 Low-Battery Pulse Blind Spot** — A known maximum detection
  delay of 175 seconds exists if a critical escalation begins immediately
  after a 5-second sampling burst concludes. Intentional tradeoff:
  preserving power for a Tier 3 "Last Gasp" SOS transmission outweighs
  the latency cost of sparse polling.

- **Endpoint Authentication & Authorization Model** — The current implementation
  is designed as an isolated edge-prototype demonstration operating on a trusted local network.
  Endpoints (`/api/telemetry/ingest`, `/api/alerts`, `/api/companion`, `/ws/risk`) enforce
  patient consent records and parameter scoping, but do not yet integrate OAuth2/JWT
  bearer authentication or mTLS. Production clinical deployments require reverse-proxy
  TLS termination and cryptographic identity validation.

- **State Machine TTL & Record Pruning** — In-memory and SQLite-backed state machines
  (PDR hysteresis, cell dismissal quarantine multipliers) maintain state across the demo
  lifecycle. Long-term production operations require automated TTL expiration sweeps and
  background pruning of historic high-frequency GPS coordinate buffers.
