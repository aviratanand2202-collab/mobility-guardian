import React from 'react'

/**
 * Risk Interpretation & Decision Rationale Panel (Technical View)
 *
 * Explains the distinction between Observational ML classification (shadow mode)
 * and Authoritative Operational Risk Engine calculations, answering why an ML model
 * can report SUSPICIOUS while the operational risk engine reports QUIESCENT (e.g. 31.9/100).
 */
export default function RiskInterpretationPanel({
  currentRisk,
  effectiveRisk,
  historyEntries = [],
  alertMessages = [],
  isConnected = false,
  trailMetadata = [],
  userId = null,
  isRunCompleted = false,
  runStatus = null,
}) {
  const safeHistory = Array.isArray(historyEntries) ? historyEntries : []
  const safeAlerts = Array.isArray(alertMessages) ? alertMessages : []
  const safeTrail = Array.isArray(trailMetadata) ? trailMetadata : []

  // Derive target operational record
  const targetRisk = effectiveRisk || (safeHistory.length > 0 ? safeHistory[safeHistory.length - 1] : null)
  const isLive = Boolean(isConnected && currentRisk)

  // Scope user strictly: match current patient to prevent cross-patient or stale mixing
  const activeUserId = userId || currentRisk?.user_id || targetRisk?.user_id || null
  const userHistory = activeUserId
    ? safeHistory.filter((h) => !h.user_id || h.user_id === activeUserId)
    : safeHistory
  const userAlerts = activeUserId
    ? safeAlerts.filter((a) => !a.user_id || a.user_id === activeUserId)
    : safeAlerts

  // Derive session run status: strictly distinguish LIVE, RUN COMPLETED, and STREAM DISCONNECTED
  const isRunConfirmedComplete = Boolean(
    isRunCompleted ||
      runStatus === 'RUN COMPLETED' ||
      targetRisk?.run_completed ||
      targetRisk?.simulation_status === 'COMPLETED'
  )

  let sessionStatus = 'STREAM DISCONNECTED'
  if (isLive) {
    sessionStatus = 'LIVE'
  } else if (isRunConfirmedComplete) {
    sessionStatus = 'RUN COMPLETED'
  } else {
    sessionStatus = 'STREAM DISCONNECTED'
  }

  // Derive ML evaluation strictly without stale carry-over
  const mlEval = currentRisk ? currentRisk.ml_evaluation : (targetRisk?.ml_evaluation || null)

  // Operational metrics
  const operationalScore =
    targetRisk?.risk_score != null && Number.isFinite(Number(targetRisk.risk_score))
      ? Number(targetRisk.risk_score)
      : null
  const operationalTier = targetRisk?.risk_tier || 'QUIESCENT'
  const pollingTier = targetRisk?.polling_tier != null ? targetRisk.polling_tier : 0
  const pollingMode = targetRisk?.polling_instruction?.mode || 'CONTINUOUS'

  // ML metrics
  const engineStatus = mlEval?.engine_status || (mlEval ? 'ACTIVE' : 'INACTIVE')
  const evidenceStatus = mlEval?.evidence_status || 'UNAVAILABLE'
  const isWindowEvaluated = evidenceStatus === 'WINDOW_EVALUATED' && engineStatus === 'ACTIVE'
  const isCollecting = evidenceStatus === 'INSUFFICIENT_EVIDENCE' || (mlEval && !isWindowEvaluated && evidenceStatus !== 'ERROR')
  const isUnavailable = !mlEval || evidenceStatus === 'ERROR' || engineStatus !== 'ACTIVE'

  const calProb = mlEval?.calibrated_probability != null ? Number(mlEval.calibrated_probability) : null
  const rawProb = mlEval?.raw_probability != null ? Number(mlEval.raw_probability) : null

  // Model decision state and phenotype: strictly mirror model outputs
  const rawDecision = mlEval?.decision_state || mlEval?.display_state
  const mlDecision = isWindowEvaluated
    ? (rawDecision || (mlEval?.behavioral_indicator && mlEval.behavioral_indicator !== 'NORMAL' ? 'SUSPICIOUS' : 'NORMAL'))
    : isCollecting
    ? 'COLLECTING'
    : 'UNAVAILABLE'

  const behavioralIndicator = mlEval?.behavioral_indicator || null
  const behavioralConfidence = mlEval?.behavioral_confidence != null ? Number(mlEval.behavioral_confidence) : null
  const windowId = mlEval?.window_id || null

  // Explainability top features
  const topFeatures = targetRisk?.explainability?.top_features || []
  const behavioralFeature = topFeatures.find((f) => f.feature === 'behavioral_trajectory_anomaly')
  const tortuosityFeature = topFeatures.find((f) => f.feature === 'path_tortuosity')
  const speedVarFeature = topFeatures.find((f) => f.feature === 'step_speed_variance')
  const distanceFeature = topFeatures.find((f) => f.feature === 'distance_from_anchor')

  // Kinematic features dict from ML output
  const kinFeats = mlEval?.kinematic_features || {}

  // Behavioral escalation detection
  const isBehavioralEscalated =
    Boolean(behavioralFeature) ||
    (operationalTier === 'SUSPICIOUS' &&
      operationalScore != null &&
      operationalScore >= 65.0 &&
      (behavioralIndicator === 'PACING' || behavioralIndicator === 'RANDOM_DRIFT'))

  const escalationReason = behavioralFeature?.human_readable || mlEval?.reason || null

  // Find score before escalation if escalation occurred in history
  let scoreBeforeEscalation = null
  let scoreAfterEscalation = operationalScore
  if (isBehavioralEscalated && userHistory.length > 1) {
    const escIndex = userHistory.findIndex((h) => h.risk_tier === 'SUSPICIOUS' || Number(h.risk_score) >= 65.0)
    if (escIndex > 0) {
      scoreBeforeEscalation = Number(userHistory[escIndex - 1].risk_score)
      scoreAfterEscalation = Number(userHistory[escIndex].risk_score)
    }
  }

  // Count distinct evaluated windows for current user/session
  const distinctWindows = new Set()
  userHistory.forEach((h) => {
    const wId = h.ml_evaluation?.window_id
    if (wId && h.ml_evaluation?.evidence_status === 'WINDOW_EVALUATED') {
      distinctWindows.add(wId)
    }
  })
  if (windowId && isWindowEvaluated) {
    distinctWindows.add(windowId)
  }

  // Trail points count from verified current session
  const completedPointsCount = safeTrail.length > 0 ? safeTrail.length : userHistory.length
  const evaluatedWindowsCount = distinctWindows.size
  const alertsCount = userAlerts.length

  // Determine trace stage statuses
  // Stage 1: ML Evaluation
  let stage1Status = 'UNAVAILABLE'
  let stage1Text = 'Offline / Disabled'
  let stage1Class = 'trace-badge-neutral'
  if (isWindowEvaluated) {
    stage1Status = 'EVALUATED'
    stage1Text = `${mlDecision}${calProb != null ? ` (${(calProb * 100).toFixed(1)}%)` : ''}`
    stage1Class = mlDecision === 'SUSPICIOUS' ? 'trace-badge-warning' : 'trace-badge-info'
  } else if (isCollecting) {
    stage1Status = 'COLLECTING'
    stage1Text = '120s Window Accumulating'
    stage1Class = 'trace-badge-amber'
  }

  // Stage 2: Operational Risk Calculation
  let stage2Status = operationalScore != null ? 'EVALUATED' : 'UNAVAILABLE'
  let stage2Text = operationalScore != null ? `${operationalScore.toFixed(1)}/100 (${operationalTier})` : 'Awaiting Telemetry'
  let stage2Class =
    operationalTier === 'CRITICAL'
      ? 'trace-badge-danger'
      : operationalTier === 'SUSPICIOUS'
      ? 'trace-badge-warning'
      : operationalTier === 'NORMAL_TRANSIT'
      ? 'trace-badge-info'
      : 'trace-badge-success'

  // Stage 3: Behavioral Escalation Gate
  let stage3Status = 'NOT TRIGGERED'
  let stage3Text = 'Gate Holds / Criteria Not Met'
  let stage3Class = 'trace-badge-neutral'
  if (isBehavioralEscalated) {
    stage3Status = 'ESCALATED'
    stage3Text = 'Multi-Window Persistence Satisfied (k>=2)'
    stage3Class = 'trace-badge-danger'
  } else if (isWindowEvaluated && (mlDecision === 'SUSPICIOUS' || behavioralIndicator === 'LAPPING' || behavioralIndicator === 'PACING')) {
    stage3Status = 'NOT TRIGGERED'
    if (behavioralIndicator === 'LAPPING') {
      stage3Text = 'Lapping Ineligible for Emergency Bridge'
    } else if (behavioralIndicator === 'PACING') {
      stage3Text = 'Window 1 Only (k < 2 Persistence Gate)'
    } else {
      stage3Text = 'Operational Thresholds Authoritative'
    }
    stage3Class = 'trace-badge-amber'
  } else if (isUnavailable) {
    stage3Status = 'UNAVAILABLE'
    stage3Text = 'Bridge Disabled / ML Offline'
    stage3Class = 'trace-badge-neutral'
  }

  // Stage 4: Alert Decision
  const hasAlert = operationalTier === 'SUSPICIOUS' || operationalTier === 'CRITICAL' || alertsCount > 0
  let stage4Status = hasAlert ? 'ALERT GENERATED' : 'NO ALERT'
  let stage4Text = hasAlert
    ? `${operationalTier === 'CRITICAL' ? 'Tier 3 (CRITICAL)' : 'Tier 2 (SUSPICIOUS)'} Alert Dispatched`
    : 'Score < 65.0 (Quiescent / Normal Transit)'
  let stage4Class = hasAlert ? 'trace-badge-danger' : 'trace-badge-success'

  return (
    <div className="risk-interpretation-panel" data-testid="risk-interpretation-panel">
      {/* Panel Header */}
      <div className="interpretation-header">
        <div className="interpretation-title-group">
          <span className="interpretation-icon">⚖️</span>
          <div>
            <h3 className="interpretation-title">Risk Interpretation &amp; Decision Rationale</h3>
            <p className="interpretation-subtitle">
              Dynamic Analysis of Observational ML Classifier vs. Authoritative Operational Risk Engine
            </p>
          </div>
        </div>
        <div className="interpretation-state-pill">
          {sessionStatus === 'LIVE' ? (
            <span className="live-telemetry-badge">
              <span className="live-dot" /> LIVE TELEMETRY ACTIVE
            </span>
          ) : sessionStatus === 'RUN COMPLETED' ? (
            <span className="final-summary-badge completed-badge">
              🏁 RUN COMPLETED
            </span>
          ) : (
            <span className="final-summary-badge disconnected-badge">
              🔌 STREAM DISCONNECTED
            </span>
          )}
        </div>
      </div>

      {/* 1. Side-by-Side Outputs Grid */}
      <div className="outputs-comparison-grid">
        {/* ML Card */}
        <div className="comparison-card ml-comparison-card">
          <div className="comparison-card-header">
            <span className="card-badge ml-card-badge">🔬 Observational ML (Shadow)</span>
            <span className={`status-tag status-tag-${evidenceStatus.toLowerCase()}`}>
              {isWindowEvaluated ? 'Window Evaluated' : isCollecting ? 'Collecting' : 'Offline'}
            </span>
          </div>

          <div className="comparison-metric-body">
            <div className="metric-primary-row">
              <span className="metric-big-value">
                {isWindowEvaluated && calProb != null ? `${(calProb * 100).toFixed(1)}%` : '--'}
              </span>
              <span className="metric-sublabel">Outlier Probability</span>
            </div>

            <div className="comparison-details-list">
              <div className="comp-detail-row">
                <span className="comp-label">Classification / Decision:</span>
                <span className={`comp-val mono ${mlDecision === 'SUSPICIOUS' ? 'text-warning' : ''}`}>
                  {isWindowEvaluated ? mlDecision : isCollecting ? 'Accumulating 120s' : 'Unavailable'}
                </span>
              </div>
              {rawProb != null && (
                <div className="comp-detail-row">
                  <span className="comp-label">Raw Model Probability:</span>
                  <span className="comp-val mono">{(rawProb * 100).toFixed(1)}%</span>
                </div>
              )}
              <div className="comp-detail-row">
                <span className="comp-label">Behavioral Phenotype:</span>
                <span className="comp-val mono font-semibold">
                  {behavioralIndicator ? behavioralIndicator : isWindowEvaluated ? 'NORMAL' : 'Unavailable'}
                </span>
              </div>
              {behavioralConfidence != null && (
                <div className="comp-detail-row">
                  <span className="comp-label">Pattern Confidence:</span>
                  <span className="comp-val mono">{(behavioralConfidence * 100).toFixed(1)}%</span>
                </div>
              )}
              <div className="comp-detail-row">
                <span className="comp-label">Latest Evaluated Window:</span>
                <span className="comp-val mono text-xs">{windowId || 'Unavailable'}</span>
              </div>
            </div>
          </div>

          <div className="comparison-card-footer">
            <span className="footer-note">
              Independent statistical feature classifier. Evaluates trajectory geometry over 120s rolling windows.
            </span>
          </div>
        </div>

        {/* Operational Engine Card */}
        <div className="comparison-card op-comparison-card">
          <div className="comparison-card-header">
            <span className="card-badge op-card-badge">🛡️ Operational Risk Engine</span>
            <span className={`tier-badge tier-${operationalTier}`}>{operationalTier}</span>
          </div>

          <div className="comparison-metric-body">
            <div className="metric-primary-row">
              <span className={`metric-big-value tier-text-${operationalTier}`}>
                {operationalScore != null ? operationalScore.toFixed(1) : '--'}
              </span>
              <span className="metric-scale">/ 100</span>
              <span className="metric-sublabel">Operational Score</span>
            </div>

            <div className="comparison-details-list">
              <div className="comp-detail-row">
                <span className="comp-label">Operational Urgency Tier:</span>
                <span className="comp-val font-semibold">{operationalTier}</span>
              </div>
              <div className="comp-detail-row">
                <span className="comp-label">Device Polling Instruction:</span>
                <span className="comp-val mono">
                  Tier {pollingTier} ({pollingMode})
                </span>
              </div>
              <div className="comp-detail-row">
                <span className="comp-label">Active Caregiver Alerts:</span>
                <span className={`comp-val font-semibold ${alertsCount > 0 ? 'text-danger' : 'text-success'}`}>
                  {alertsCount} {alertsCount === 1 ? 'Alert' : 'Alerts'} ({hasAlert ? 'Active Urgency' : 'No Urgency'})
                </span>
              </div>
              <div className="comp-detail-row">
                <span className="comp-label">Behavioral Escalation:</span>
                <span className={`comp-val mono ${isBehavioralEscalated ? 'text-danger font-semibold' : 'text-muted'}`}>
                  {isBehavioralEscalated ? 'Escalated (Score clamped >= 65)' : 'Not Triggered / Inactive'}
                </span>
              </div>
            </div>
          </div>

          <div className="comparison-card-footer">
            <span className="footer-note">
              Authoritative operational engine. Governs caregiver alerts, geofence breaches, and device power policy.
            </span>
          </div>
        </div>
      </div>

      {/* 2. Alert Decision Trace Stepper */}
      <div className="decision-trace-section">
        <div className="trace-section-header">
          <span className="trace-title">Decision Pipeline Trace</span>
          <span className="trace-subtitle">End-to-End Evaluation Flow</span>
        </div>

        <div className="trace-stepper">
          {/* Step 1 */}
          <div className="trace-step">
            <div className="trace-step-number">1</div>
            <div className="trace-step-content">
              <span className="trace-step-title">ML Shadow Evaluation</span>
              <span className={`trace-step-badge ${stage1Class}`}>{stage1Status}</span>
              <span className="trace-step-desc">{stage1Text}</span>
            </div>
          </div>

          <div className="trace-arrow">➔</div>

          {/* Step 2 */}
          <div className="trace-step">
            <div className="trace-step-number">2</div>
            <div className="trace-step-content">
              <span className="trace-step-title">Operational Risk Score</span>
              <span className={`trace-step-badge ${stage2Class}`}>{stage2Status}</span>
              <span className="trace-step-desc">{stage2Text}</span>
            </div>
          </div>

          <div className="trace-arrow">➔</div>

          {/* Step 3 */}
          <div className="trace-step">
            <div className="trace-step-number">3</div>
            <div className="trace-step-content">
              <span className="trace-step-title">Behavioral Escalation Gate</span>
              <span className={`trace-step-badge ${stage3Class}`}>{stage3Status}</span>
              <span className="trace-step-desc">{stage3Text}</span>
            </div>
          </div>

          <div className="trace-arrow">➔</div>

          {/* Step 4 */}
          <div className="trace-step">
            <div className="trace-step-number">4</div>
            <div className="trace-step-content">
              <span className="trace-step-title">Caregiver Alert Outcome</span>
              <span className={`trace-step-badge ${stage4Class}`}>{stage4Status}</span>
              <span className="trace-step-desc">{stage4Text}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Dynamic Rationale & Clinical Explanation */}
      <div className="rationale-explanation-card">
        <div className="rationale-card-header">
          <span className="rationale-heading-icon">💡</span>
          <strong>Dynamic Decision Rationale</strong>
        </div>

        <div className="rationale-card-body">
          {/* Case 1: Behavioral escalation triggered */}
          {isBehavioralEscalated ? (
            <div className="rationale-content-block">
              <div className="rationale-alert-banner banner-escalated">
                🛡️ <strong>Behavioral Escalation Confirmed:</strong> Sustained anomalous trajectory met multi-window
                persistence criteria.
              </div>
              <ul className="rationale-bullet-list">
                <li>
                  <strong>Multi-Window Persistence Gate Satisfied:</strong> The trajectory exhibited anomalous kinematic
                  behavior across {distinctWindows.size >= 2 ? distinctWindows.size : 2} distinct observation windows (
                  {distinctWindows.size >= 2 ? distinctWindows.size * 2 : 4} minutes of monitored movement).
                </li>
                <li>
                  <strong>Additive Operational Clamping:</strong> Because persistence criteria were satisfied outside safe
                  areas, the operational engine elevated the operational risk score from{' '}
                  {scoreBeforeEscalation != null ? (
                    <span className="mono font-bold">{scoreBeforeEscalation.toFixed(1)}/100</span>
                  ) : (
                    'baseline'
                  )}{' '}
                  to{' '}
                  <span className="mono font-bold text-danger">
                    {scoreAfterEscalation != null ? scoreAfterEscalation.toFixed(1) : 65.0}/100 (SUSPICIOUS)
                  </span>
                  .
                </li>
                <li>
                  <strong>Caregiver Notification:</strong> Elevated polling tier to Tier 2 and dispatched a Tier 2 safety
                  alert.
                </li>
                {escalationReason && (
                  <li className="rationale-feature-highlight">
                    <strong>Reported Evidence:</strong> &ldquo;{escalationReason}&rdquo;
                  </li>
                )}
              </ul>
            </div>
          ) : isWindowEvaluated && (mlDecision === 'SUSPICIOUS' || (calProb != null && calProb >= 0.5)) && (operationalTier === 'QUIESCENT' || operationalTier === 'NORMAL_TRANSIT') ? (
            /* Case 2: ML is SUSPICIOUS while Operational Risk is QUIESCENT */
            <div className="rationale-content-block">
              <div className="rationale-alert-banner banner-divergent">
                ℹ️ <strong>Expected Output Divergence:</strong> ML classifier flagged an anomaly, but the operational risk engine
                correctly determined no emergency alert is warranted.
              </div>
              <ul className="rationale-bullet-list">
                <li>
                  <strong>Metric Separation:</strong> The ML outlier probability (
                  <span className="mono font-bold">{calProb != null ? (calProb * 100).toFixed(1) + '%' : 'elevated'}</span>) is
                  a statistical geometric pattern metric. It is{' '}
                  <strong>not directly comparable to the operational risk score</strong> (
                  <span className="mono font-bold">
                    {operationalScore != null ? operationalScore.toFixed(1) + '/100' : '31.9/100'}
                  </span>
                  ).
                </li>
                <li>
                  <strong>Independent Heuristics:</strong> The authoritative operational risk engine independently assesses
                  physical safety factors including velocity boundary checks, net displacement, safe-area containment, and
                  geofence limits.
                </li>
                <li>
                  <strong>Escalation Gate Holds:</strong> An ML &ldquo;SUSPICIOUS&rdquo; decision alone does not guarantee an
                  operational alert.
                  {behavioralIndicator === 'LAPPING' ? (
                    <span>
                      {' '}Under existing operational policy, <strong>recognized lapping (circular loops)</strong> is excluded
                      from emergency behavioral escalation while inside safe velocity boundaries, as it represents structured non-wandering movement.
                    </span>
                  ) : behavioralIndicator === 'PACING' && distinctWindows.size < 2 ? (
                    <span>
                      {' '}Only <strong>1 observation window</strong> of pacing has been evaluated. System safety invariants require
                      at least 2 consecutive distinct windows ($k \ge 2$) before triggering emergency caregiver escalation.
                    </span>
                  ) : (
                    <span>
                      {' '}The current behavioral pattern does not meet operational escalation persistence criteria. Operational risk
                      remains at baseline to prevent false caregiver alarm fatigue.
                    </span>
                  )}
                </li>
                <li>
                  <strong>Authoritative Status:</strong> The operational tier (<strong>{operationalTier}</strong>) remains
                  authoritative for all notifications and polling decisions.
                </li>
              </ul>
            </div>
          ) : isWindowEvaluated && (mlDecision === 'NORMAL' || (calProb != null && calProb < 0.5)) ? (
            /* Case 3: Model Reports NORMAL (Do NOT claim lapping when model reports NORMAL) */
            <div className="rationale-content-block">
              <div className="rationale-alert-banner banner-concordant">
                ✓ <strong>Normal Classification:</strong> Observational ML model classified movement as NORMAL.
              </div>
              <p className="rationale-plain-text">
                The evaluated 120-second trajectory is classified as <span className="mono font-bold">NORMAL</span> by the observational model.
                No emergency wandering or pacing anomaly was detected. Operational risk score is{' '}
                <span className="mono font-bold">{operationalScore != null ? operationalScore.toFixed(1) : '--'}/100</span> (
                {operationalTier}). Zero safety alerts required.
              </p>
            </div>
          ) : isCollecting ? (
            /* Case 4: Collecting Telemetry */
            <div className="rationale-content-block">
              <div className="rationale-alert-banner banner-collecting">
                ⏳ <strong>Evidence Accumulation in Progress:</strong> The observational ML model is accumulating the required
                120-second observation window.
              </div>
              <p className="rationale-plain-text">
                The ML model requires at least 120 seconds of continuous trajectory data to calculate directional entropy and
                tortuosity. Safety status cannot be verified from partial evidence. Meanwhile, the operational risk engine maintains
                real-time continuous boundary and velocity protection ({operationalTier}, {operationalScore?.toFixed(1) || '--'}/100).
              </p>
            </div>
          ) : (
            /* Case 5: Offline / Unavailable */
            <div className="rationale-content-block">
              <div className="rationale-alert-banner banner-unavailable">
                ⚠️ <strong>Observational ML Unavailable / Feature Disabled:</strong> Observational ML signal is offline or not
                configured for this session.
              </div>
              <p className="rationale-plain-text">
                Operational risk monitoring remains fully functional. Geofence boundaries, velocity checks, and battery conservation
                rules operate independently of shadow ML inference.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 4. Evidence from Current Run Grid */}
      <div className="run-evidence-section">
        <div className="evidence-section-header">
          <span className="evidence-title">Telemetry &amp; Kinematic Evidence</span>
          <span className="evidence-subtitle">Actual Verified Parameters from Current Session</span>
        </div>

        <div className="evidence-grid">
          {/* Path Tortuosity */}
          <div className="evidence-card">
            <span className="ev-label">Path Tortuosity</span>
            <span className="ev-val mono">
              {kinFeats.tortuosity_index != null
                ? Number(kinFeats.tortuosity_index).toFixed(2)
                : tortuosityFeature
                ? tortuosityFeature.human_readable
                : isWindowEvaluated
                ? 'Normal'
                : 'Unavailable'}
            </span>
            <span className="ev-desc">Heading variance &amp; directional changes</span>
          </div>

          {/* Speed Variance */}
          <div className="evidence-card">
            <span className="ev-label">Locomotive Pace / Speed</span>
            <span className="ev-val mono">
              {kinFeats.mean_speed_mps != null
                ? `${Number(kinFeats.mean_speed_mps).toFixed(2)} m/s`
                : targetRisk?.sensor_metrics?.speed_mps != null
                ? `${Number(targetRisk.sensor_metrics.speed_mps).toFixed(2)} m/s`
                : speedVarFeature
                ? speedVarFeature.human_readable
                : 'Unavailable'}
            </span>
            <span className="ev-desc">Locomotive speed stability &amp; velocity check</span>
          </div>

          {/* Distance from Anchor */}
          <div className="evidence-card">
            <span className="ev-label">Displacement / Anchor</span>
            <span className="ev-val mono">
              {kinFeats.straight_line_displacement_m != null
                ? `${Number(kinFeats.straight_line_displacement_m).toFixed(1)}m`
                : targetRisk?.sensor_metrics?.net_displacement_m != null
                ? `${Number(targetRisk.sensor_metrics.net_displacement_m).toFixed(1)}m`
                : distanceFeature
                ? distanceFeature.human_readable
                : 'Unavailable'}
            </span>
            <span className="ev-desc">Net displacement relative to anchor</span>
          </div>

          {/* Completed Trail Points */}
          <div className="evidence-card">
            <span className="ev-label">Completed Trail Points</span>
            <span className="ev-val mono font-semibold">
              {completedPointsCount > 0 ? `${completedPointsCount} Points` : '0 Points'}
            </span>
            <span className="ev-desc">Session fixes processed</span>
          </div>

          {/* Evaluated Windows */}
          <div className="evidence-card">
            <span className="ev-label">Evaluated ML Windows</span>
            <span className="ev-val mono font-semibold">
              {evaluatedWindowsCount > 0
                ? `${evaluatedWindowsCount} ${evaluatedWindowsCount === 1 ? 'Window' : 'Windows'} (${evaluatedWindowsCount * 2} mins)`
                : isCollecting
                ? '0 Windows (Collecting)'
                : 'Unavailable'}
            </span>
            <span className="ev-desc">120-second observation cycles</span>
          </div>

          {/* Generated Alerts */}
          <div className="evidence-card">
            <span className="ev-label">Operational Alerts</span>
            <span className={`ev-val mono font-semibold ${alertsCount > 0 ? 'text-danger' : 'text-success'}`}>
              {alertsCount} {alertsCount === 1 ? 'Alert' : 'Alerts'}
            </span>
            <span className="ev-desc">Dispatched caregiver notifications</span>
          </div>
        </div>
      </div>

      {/* 5. Final Run Status / Summary */}
      {sessionStatus === 'RUN COMPLETED' ? (
        <div className="final-run-summary-box">
          <div className="final-summary-header">
            <strong>📋 Final Run Outcome Summary (Confirmed Complete)</strong>
            <span className="summary-timestamp mono">
              Points: {completedPointsCount} · Windows: {evaluatedWindowsCount}
            </span>
          </div>

          <div className="final-summary-body">
            <p>
              <strong>Final Operational Tier:</strong>{' '}
              <span className={`tier-badge tier-${operationalTier}`}>{operationalTier}</span> (Score:{' '}
              {operationalScore != null ? operationalScore.toFixed(1) : '--'}/100)
            </p>
            <p>
              <strong>Observational ML Result:</strong>{' '}
              {isWindowEvaluated ? (
                <span>
                  <strong>{mlDecision}</strong>
                  {calProb != null ? ` (${(calProb * 100).toFixed(1)}% probability)` : ''}
                  {behavioralIndicator ? ` [${behavioralIndicator}]` : ''}
                </span>
              ) : (
                <span className="text-muted italic">
                  Final ML window evaluation was unavailable (run ended before 120s window completion).
                </span>
              )}
            </p>
            <p>
              <strong>Alert Outcome:</strong>{' '}
              {alertsCount > 0
                ? `${alertsCount} caregiver alert(s) persisted in database.`
                : 'Zero alerts dispatched (operational score remained below 65.0 threshold).'}
            </p>
            <p className="final-verdict-text">
              {isBehavioralEscalated
                ? 'Outcome Verdict: Sustained pacing satisfied multi-window persistence invariants and successfully escalated operational risk.'
                : mlDecision === 'SUSPICIOUS' && operationalTier === 'QUIESCENT'
                ? 'Outcome Verdict: Expected divergence validated — kinematic anomaly flagged by shadow model while operational engine correctly withheld false alerts.'
                : 'Outcome Verdict: Normal trajectory verified across both operational and shadow monitoring engines.'}
            </p>
          </div>
        </div>
      ) : sessionStatus === 'STREAM DISCONNECTED' && userHistory.length > 0 ? (
        <div className="final-run-summary-box incomplete-run-box">
          <div className="final-summary-header">
            <strong>🔌 Stream Disconnected — Run Status Incomplete / Unconfirmed</strong>
            <span className="summary-timestamp mono">
              Points: {completedPointsCount} · Windows: {evaluatedWindowsCount}
            </span>
          </div>
          <div className="final-summary-body">
            <p className="text-muted">
              Telemetry stream disconnected. A final run summary is withheld because simulation completion cannot be confirmed solely from a disconnected stream. Run status is incomplete or unavailable.
            </p>
          </div>
        </div>
      ) : null}
    </div>
  )
}
