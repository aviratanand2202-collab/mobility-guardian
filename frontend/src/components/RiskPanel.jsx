import React from 'react'

export default function RiskPanel({ currentRisk, historyEntries }) {
  const riskTier = currentRisk?.risk_tier || 'QUIESCENT'
  const riskScore = currentRisk?.risk_score != null ? Number(currentRisk.risk_score) : null
  const hasValidScore = typeof riskScore === 'number' && Number.isFinite(riskScore)
  const pollingMode = currentRisk?.polling_instruction?.mode || 'CONTINUOUS'
  const pollingTier = currentRisk?.polling_tier != null ? currentRisk.polling_tier : '-'
  const leadTimeSec = currentRisk?.predicted_lead_time_sec

  // Extract explainability top features
  const topFeatures = currentRisk?.explainability?.top_features || []

  // Prepare trend data: take the most recent 15 points (oldest first -> newest last),
  // plus include currentRisk if newer / not already represented
  const points = [...historyEntries].slice(-15)
  if (currentRisk && (points.length === 0 || points[points.length - 1].timestamp !== currentRisk.timestamp)) {
    points.push({
      timestamp: currentRisk.timestamp,
      risk_score: currentRisk.risk_score,
      risk_tier: currentRisk.risk_tier,
    })
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
    const clamped = Math.max(0, Math.min(100, score || 0))
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
            <span className="detail-val mono">{currentRisk?.battery_override_active ? 'Active' : 'Normal'}</span>
          </div>
        </div>
      </div>

      {/* Lightweight SVG Trend Line Chart */}
      <div className="trend-chart-section">
        <div className="section-label">
          <span>Risk Score Trend (Last {trendPoints.length} Points)</span>
          {trendPoints.length > 0 && (
            <span className="mono" style={{ fontSize: '11px', color: '#6b7280' }}>
              Latest: {trendPoints[trendPoints.length - 1].risk_score?.toFixed(1) || '-'}
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
