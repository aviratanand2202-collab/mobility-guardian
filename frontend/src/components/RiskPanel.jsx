import React from 'react'

export default function RiskPanel({ currentRisk, historyEntries = [] }) {
  // Derive effective risk evaluation: prefer live WebSocket currentRisk;
  // fall back to the most recent record from historyEntries when connected with existing history
  const safeHistory = Array.isArray(historyEntries) ? historyEntries : []
  const latestHistory = safeHistory.length > 0
    ? safeHistory[safeHistory.length - 1]
    : null

  // Determine if live score is a valid finite number
  const isLiveScoreValid =
    currentRisk?.risk_score != null &&
    typeof currentRisk.risk_score !== 'boolean' &&
    currentRisk.risk_score !== '' &&
    Number.isFinite(Number(currentRisk.risk_score))

  // Determine effective risk record:
  // Prefer live currentRisk when it contains a valid score; otherwise fall back to latest accepted history
  const effectiveRisk = isLiveScoreValid ? currentRisk : (latestHistory || currentRisk)

  const riskTier = effectiveRisk?.risk_tier || 'QUIESCENT'
  const riskScoreRaw = effectiveRisk?.risk_score
  const hasValidScore =
    riskScoreRaw != null &&
    typeof riskScoreRaw !== 'boolean' &&
    riskScoreRaw !== '' &&
    Number.isFinite(Number(riskScoreRaw))
  const riskScore = hasValidScore ? Number(riskScoreRaw) : null

  const pollingMode = effectiveRisk?.polling_instruction?.mode || 'CONTINUOUS'
  const pollingTier = effectiveRisk?.polling_tier != null ? effectiveRisk.polling_tier : '-'
  const leadTimeSec = effectiveRisk?.predicted_lead_time_sec

  // Extract explainability top features
  const topFeatures = effectiveRisk?.explainability?.top_features || []

  // Derive observational shadow ML evaluation:
  // Prefer live WebSocket currentRisk.ml_evaluation; when currentRisk is present,
  // do not fall back to previous windows or history entries (strictly prevents stale carry-over).
  const mlEvaluation = currentRisk ? currentRisk.ml_evaluation : latestHistory?.ml_evaluation

  // Prepare trend data: take the most recent 15 points (oldest first -> newest last),
  // plus include currentRisk if newer / not already represented
  const points = [...safeHistory].slice(-15)
  if (currentRisk && (points.length === 0 || points[points.length - 1].timestamp !== currentRisk.timestamp)) {
    if (isLiveScoreValid) {
      points.push({
        timestamp: currentRisk.timestamp,
        risk_score: Number(currentRisk.risk_score),
        risk_tier: currentRisk.risk_tier,
      })
    }
  }
  const trendPoints = points.slice(-15)

  // SVG Trendline calculations
  const svgWidth = 320
  const svgHeight = 90
  const padLeft = 24
  const padRight = 10
  const padTop = 10
  const padBottom = 16
  const plotW = svgWidth - padLeft - padRight
  const plotH = svgHeight - padTop - padBottom

  const getY = (score) => {
    const num = Number(score)
    const clamped = Math.max(0, Math.min(100, Number.isFinite(num) ? num : 0))
    return padTop + plotH - (clamped / 100) * plotH
  }

  const getX = (index, total) => {
    if (total <= 1) return padLeft + plotW / 2
    return padLeft + (index / (total - 1)) * plotW
  }

  const polylineCoords = trendPoints.map((p, idx) => {
    const x = getX(idx, trendPoints.length)
    const y = getY(p.risk_score)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <div className="panel risk-panel-container">
      <div className="panel-header">
        <h2 className="panel-title">Risk Engine &amp; Explainability</h2>
        <span className={`tier-badge tier-${riskTier}`}>
          {riskTier}
        </span>
      </div>

      {/* Prominent Score Card */}
      <div className="risk-metric-card">
        <div className="metric-score-block">
          <div className="metric-label">Current Risk Score</div>
          <div className="metric-value-row">
            {hasValidScore ? (
              <>
                <span className={`metric-value tier-text-${riskTier}`}>
                  {riskScore.toFixed(1)}
                </span>
                <span className="metric-scale">/ 100</span>
              </>
            ) : (
              <>
                <span
                  className="metric-value metric-placeholder"
                  style={{ fontWeight: 500, color: '#9ca3af', letterSpacing: '2px' }}
                >
                  --
                </span>
                <span className="metric-scale">/ 100</span>
              </>
            )}
          </div>
          {leadTimeSec != null && (
            <div className="lead-time-notice">
              Est. Breach Lead Time: <strong>{leadTimeSec}s</strong>
            </div>
          )}
        </div>

        <div className="metric-details-block">
          <div className="detail-item">
            <span className="detail-label">Polling Mode:</span>
            <span className="detail-val mono">{pollingMode}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Polling Tier:</span>
            <span className="detail-val mono">Tier {pollingTier}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Battery Override:</span>
            <span className="detail-val mono">{effectiveRisk?.battery_override_active ? 'Active' : 'Normal'}</span>
          </div>
        </div>
      </div>

      {/* Lightweight SVG Trend Line Chart */}
      <div className="trend-chart-section">
        <div className="section-label">
          <span>Risk Score Trend (Last {trendPoints.length} Points)</span>
          {trendPoints.length > 0 && (
            <span className="mono" style={{ fontSize: '11px', color: '#6b7280' }}>
              Latest: {
                trendPoints[trendPoints.length - 1].risk_score != null &&
                Number.isFinite(Number(trendPoints[trendPoints.length - 1].risk_score))
                  ? Number(trendPoints[trendPoints.length - 1].risk_score).toFixed(1)
                  : '-'
              }
            </span>
          )}
        </div>

        {trendPoints.length === 0 ? (
          <div className="chart-empty">No score points recorded yet.</div>
        ) : (
          <div className="svg-container">
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="trend-svg"
              preserveAspectRatio="none"
            >
              {/* Reference Threshold Lines */}
              {/* Tier 1: 35 (NORMAL_TRANSIT) */}
              <line
                x1={padLeft}
                y1={getY(35)}
                x2={svgWidth - padRight}
                y2={getY(35)}
                stroke="#16a34a"
                strokeWidth="1"
                strokeDasharray="2,3"
                opacity="0.4"
              />
              <text x="2" y={getY(35) + 3} fontSize="8" fill="#16a34a" opacity="0.8">35</text>

              {/* Tier 2: 65 (SUSPICIOUS) */}
              <line
                x1={padLeft}
                y1={getY(65)}
                x2={svgWidth - padRight}
                y2={getY(65)}
                stroke="#ea580c"
                strokeWidth="1"
                strokeDasharray="2,3"
                opacity="0.4"
              />
              <text x="2" y={getY(65) + 3} fontSize="8" fill="#ea580c" opacity="0.8">65</text>

              {/* Tier 3: 85 (CRITICAL) */}
              <line
                x1={padLeft}
                y1={getY(85)}
                x2={svgWidth - padRight}
                y2={getY(85)}
                stroke="#dc2626"
                strokeWidth="1"
                strokeDasharray="2,3"
                opacity="0.4"
              />
              <text x="2" y={getY(85) + 3} fontSize="8" fill="#dc2626" opacity="0.8">85</text>

              {/* Polyline Path */}
              {trendPoints.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={polylineCoords}
                />
              )}

              {/* Data Point Circles */}
              {trendPoints.map((p, idx) => {
                const cx = getX(idx, trendPoints.length)
                const cy = getY(p.risk_score)
                const isLatest = idx === trendPoints.length - 1
                return (
                  <circle
                    key={idx}
                    cx={cx}
                    cy={cy}
                    r={isLatest ? 4 : 2.5}
                    fill={isLatest ? '#dc2626' : '#2563eb'}
                    stroke="#ffffff"
                    strokeWidth={isLatest ? 1.5 : 1}
                  />
                )
              })}
            </svg>
          </div>
        )}
      </div>

      {/* Observational Shadow ML Evaluation Display (rendered only when ml_evaluation is present) */}
      {mlEvaluation != null && (
        <div className="ml-evaluation-card" data-testid="ml-evaluation-card">
          <div className="ml-eval-header">
            <span className="ml-eval-title">
              <span aria-hidden="true">🔬</span> Observational ML Evaluation (Shadow Mode)
            </span>
            <span
              className={`ml-eval-pill ml-eval-pill-${
                currentRisk && mlEvaluation.evidence_status === 'WINDOW_EVALUATED' && mlEvaluation.engine_status === 'ACTIVE'
                  ? 'evaluated'
                  : mlEvaluation.evidence_status === 'INSUFFICIENT_EVIDENCE' || !currentRisk
                  ? 'collecting'
                  : 'unavailable'
              }`}
              data-testid="ml-eval-status-pill"
            >
              {currentRisk && mlEvaluation.evidence_status === 'WINDOW_EVALUATED' && mlEvaluation.engine_status === 'ACTIVE'
                ? 'Window Evaluated'
                : mlEvaluation.evidence_status === 'INSUFFICIENT_EVIDENCE' || !currentRisk
                ? 'Collecting Telemetry'
                : 'Unavailable'}
            </span>
          </div>

          <div className="ml-eval-body">
            {currentRisk &&
            mlEvaluation.evidence_status === 'WINDOW_EVALUATED' &&
            mlEvaluation.engine_status === 'ACTIVE' &&
            mlEvaluation.calibrated_probability != null ? (
              <div className="ml-eval-metric">
                <div className="ml-eval-metric-val" data-testid="ml-eval-probability">
                  {(mlEvaluation.calibrated_probability * 100).toFixed(1)}%
                </div>
                <div className="ml-eval-metric-label">
                  Kinematic Outlier Probability (120s Window)
                </div>
              </div>
            ) : mlEvaluation.evidence_status === 'INSUFFICIENT_EVIDENCE' || !currentRisk ? (
              <div className="ml-eval-status-block">
                <div className="ml-eval-status-msg" data-testid="ml-eval-insufficient-msg">
                  Collecting telemetry
                </div>
                <div className="ml-eval-status-desc">
                  {!currentRisk
                    ? 'Awaiting live telemetry for current connection. Safety cannot be verified from incomplete evidence.'
                    : 'Accumulating 120s window. Safety cannot be verified from incomplete evidence.'}
                </div>
              </div>
            ) : (
              <div className="ml-eval-status-block">
                <div className="ml-eval-status-msg" data-testid="ml-eval-unavailable-msg">
                  ML Evaluation Unavailable
                </div>
                <div className="ml-eval-status-desc">
                  Observational signal temporarily offline. Operational risk monitoring remains active.
                </div>
              </div>
            )}

            {currentRisk && mlEvaluation.evidence_status === 'WINDOW_EVALUATED' && mlEvaluation.engine_status === 'ACTIVE' && (
              <div className="ml-eval-meta-block">
                {mlEvaluation.decision_state && (
                  <div className="detail-item">
                    <span className="detail-label">Decision:</span>
                    <span className="detail-val mono">{mlEvaluation.decision_state}</span>
                  </div>
                )}
                {mlEvaluation.raw_probability != null && (
                  <div className="detail-item">
                    <span className="detail-label">Raw Prob:</span>
                    <span className="detail-val mono">{(mlEvaluation.raw_probability * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="ml-eval-disclaimer">
            Observational ML evaluation only — not a clinical prediction, diagnosis, or operational safety decision.
          </div>
        </div>
      )}

      {/* TreeSHAP Explainability Top Features */}
      <div className="explainability-section">
        <div className="section-label">Kinematic Explainability (TreeSHAP)</div>
        {topFeatures.length === 0 ? (
          <div className="features-empty">
            No anomalous kinematic patterns flagged for this window.
          </div>
        ) : (
          <div className="features-list">
            {topFeatures.map((feat, idx) => (
              <div key={idx} className="feature-item">
                <div className="feature-header">
                  <span className="feature-text">{feat.human_readable}</span>
                  <span className="feature-shap mono">
                    SHAP: {feat.shap_value > 0 ? `+${feat.shap_value.toFixed(2)}` : feat.shap_value.toFixed(2)}
                  </span>
                </div>
                <div className="feature-raw mono">{feat.feature}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
