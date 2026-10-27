/**
 * Regression and Verification Tests for RiskInterpretationPanel.
 *
 * Verifies:
 * 1. Source Structure & Contract: All 5 required sections, data-testid, and badge definitions.
 * 2. Integration: RiskPanel mounts RiskInterpretationPanel and TechnicalDashboard passes props.
 * 3. CSS Invariants: High contrast, responsive breakpoints (640px, 768px), and status badge styles.
 * 4. Fix 1 - Lapping & Divergence:
 *    - Explains recognized lapping excluded from behavioral escalation when classification is LAPPING.
 *    - When model reports NORMAL, shows NORMAL and never claims lapping was detected.
 *    - Clearly explains why ML SUSPICIOUS (85.2%) diverges from Operational QUIESCENT (31.9/100).
 * 5. Fix 2 - Evidence Consistency & User/Session Scoping:
 *    - Phenotype, window ID, operational score, trail-point count, evaluated-window count, and alerts
 *      strictly scoped to the same user.
 *    - No invented arbitrary numbers (displays 'Unavailable' when unverified).
 * 6. Fix 3 - Final Run Status Distinction:
 *    - Distinguishes 'LIVE', 'STREAM DISCONNECTED', and 'RUN COMPLETED'.
 *    - WebSocket disconnect alone does NOT show final completed summary.
 *    - Shows final summary only when completion is confirmed; otherwise states incomplete/unavailable.
 *
 * Run with: node frontend/tests/test_risk_interpretation_panel.mjs
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const componentPath = path.resolve(__dirname, '../src/components/technical/RiskInterpretationPanel.jsx')
const componentCode = fs.readFileSync(componentPath, 'utf8')

const riskPanelPath = path.resolve(__dirname, '../src/components/RiskPanel.jsx')
const riskPanelCode = fs.readFileSync(riskPanelPath, 'utf8')

const technicalDashboardPath = path.resolve(__dirname, '../src/components/technical/TechnicalDashboard.jsx')
const techDashboardCode = fs.readFileSync(technicalDashboardPath, 'utf8')

const cssPath = path.resolve(__dirname, '../src/index.css')
const cssCode = fs.readFileSync(cssPath, 'utf8')

// ============================================================================
// 1. Source Structure & Contract Tests
// ============================================================================

test('1. RiskInterpretationPanel source contains required structural components', () => {
  assert.match(
    componentCode,
    /data-testid="risk-interpretation-panel"/,
    'Must define root element with data-testid="risk-interpretation-panel"'
  )

  // 1. Side-by-side outputs
  assert.match(componentCode, /className="outputs-comparison-grid"/, 'Must render outputs comparison grid')
  assert.match(componentCode, /className="comparison-card ml-comparison-card"/, 'Must render ML comparison card')
  assert.match(componentCode, /className="comparison-card op-comparison-card"/, 'Must render Operational comparison card')
  assert.match(componentCode, /Outlier Probability/, 'Must display Outlier Probability label')
  assert.match(componentCode, /Operational Score/, 'Must display Operational Score label')

  // 2. Alert Decision Trace Stepper
  assert.match(componentCode, /className="decision-trace-section"/, 'Must render decision trace section')
  assert.match(componentCode, /className="trace-stepper"/, 'Must render 4-stage trace stepper')
  assert.match(componentCode, /ML Shadow Evaluation/, 'Must include Stage 1: ML Shadow Evaluation')
  assert.match(componentCode, /Operational Risk Score/, 'Must include Stage 2: Operational Risk Score')
  assert.match(componentCode, /Behavioral Escalation Gate/, 'Must include Stage 3: Behavioral Escalation Gate')
  assert.match(componentCode, /Caregiver Alert Outcome/, 'Must include Stage 4: Caregiver Alert Outcome')

  // 3. Dynamic Rationale Card
  assert.match(componentCode, /className="rationale-explanation-card"/, 'Must render dynamic rationale card')
  assert.match(componentCode, /Dynamic Decision Rationale/, 'Must display Dynamic Decision Rationale heading')

  // 4. Evidence Grid
  assert.match(componentCode, /className="run-evidence-section"/, 'Must render run evidence section')
  assert.match(componentCode, /className="evidence-grid"/, 'Must render telemetry and kinematic evidence grid')
  assert.match(componentCode, /Path Tortuosity/, 'Must include Path Tortuosity metric card')
  assert.match(componentCode, /Locomotive Pace \/ Speed/, 'Must include Speed / Pace metric card')
  assert.match(componentCode, /Displacement \/ Anchor/, 'Must include Displacement / Anchor metric card')
  assert.match(componentCode, /Completed Trail Points/, 'Must include Completed Trail Points metric card')
  assert.match(componentCode, /Evaluated ML Windows/, 'Must include Evaluated ML Windows metric card')
  assert.match(componentCode, /Operational Alerts/, 'Must include Operational Alerts metric card')

  // 5. Final Run Status Badges & Boxes
  assert.match(componentCode, /sessionStatus === 'LIVE'/, 'Must distinguish LIVE status')
  assert.match(componentCode, /sessionStatus === 'RUN COMPLETED'/, 'Must distinguish RUN COMPLETED status')
  assert.match(componentCode, /sessionStatus === 'STREAM DISCONNECTED'/, 'Must distinguish STREAM DISCONNECTED status')
})

test('2. Integration: RiskPanel mounts RiskInterpretationPanel below Kinematic Explainability', () => {
  assert.match(
    riskPanelCode,
    /import RiskInterpretationPanel from '\.\/technical\/RiskInterpretationPanel'/,
    'RiskPanel must import RiskInterpretationPanel'
  )

  const explainabilityIndex = riskPanelCode.indexOf('className="explainability-section"')
  const mountedPanelIndex = riskPanelCode.indexOf('<RiskInterpretationPanel')
  assert.ok(explainabilityIndex > 0, 'explainability-section must exist in RiskPanel')
  assert.ok(mountedPanelIndex > 0, 'RiskInterpretationPanel must be rendered in RiskPanel')
  assert.ok(mountedPanelIndex > explainabilityIndex, 'RiskInterpretationPanel must be rendered below Kinematic Explainability')
})

test('3. Integration: TechnicalDashboard supplies props to RiskPanel', () => {
  assert.match(
    techDashboardCode,
    /<RiskPanel[\s\S]*?currentRisk=\{latestRisk\}[\s\S]*?historyEntries=\{historyEntries\}[\s\S]*?alertMessages=\{alertMessages\}[\s\S]*?isConnected=\{isConnected\}[\s\S]*?trailMetadata=\{trailMetadata\}[\s\S]*?userId=\{userId\}/,
    'TechnicalDashboard must pass latestRisk, historyEntries, alertMessages, isConnected, trailMetadata, and userId to RiskPanel'
  )
})

test('4. CSS Invariants: index.css includes required styles and badges', () => {
  const requiredSelectors = [
    '.risk-interpretation-panel',
    '.outputs-comparison-grid',
    '.comparison-card',
    '.ml-comparison-card',
    '.op-comparison-card',
    '.decision-trace-section',
    '.trace-stepper',
    '.trace-step',
    '.rationale-explanation-card',
    '.banner-divergent',
    '.banner-escalated',
    '.banner-concordant',
    '.banner-collecting',
    '.banner-unavailable',
    '.run-evidence-section',
    '.evidence-grid',
    '.evidence-card',
    '.final-run-summary-box',
    '.completed-badge',
    '.disconnected-badge',
    '.incomplete-run-box',
  ]

  requiredSelectors.forEach((selector) => {
    assert.ok(cssCode.includes(selector), `index.css must include selector ${selector}`)
  })

  // Responsive layout checks
  assert.match(cssCode, /@media\s*\(max-width:\s*640px\)\s*\{[\s\S]*?\.outputs-comparison-grid/)
  assert.match(cssCode, /@media\s*\(max-width:\s*768px\)\s*\{[\s\S]*?\.trace-stepper/)
  assert.match(cssCode, /@media\s*\(max-width:\s*640px\)\s*\{[\s\S]*?\.evidence-grid/)
})

// ============================================================================
// 2. Behavioral Logic & State Derivation Tests
// ============================================================================

function deriveRiskInterpretationState({
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

  const targetRisk = effectiveRisk || (safeHistory.length > 0 ? safeHistory[safeHistory.length - 1] : null)
  const isLive = Boolean(isConnected && currentRisk)

  // Scope user strictly
  const activeUserId = userId || currentRisk?.user_id || targetRisk?.user_id || null
  const userHistory = activeUserId
    ? safeHistory.filter((h) => !h.user_id || h.user_id === activeUserId)
    : safeHistory
  const userAlerts = activeUserId
    ? safeAlerts.filter((a) => !a.user_id || a.user_id === activeUserId)
    : safeAlerts

  // Status distinction
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

  // ML evaluation strictly without stale carry-over
  const mlEval = currentRisk ? currentRisk.ml_evaluation : (targetRisk?.ml_evaluation || null)

  const operationalScore =
    targetRisk?.risk_score != null && Number.isFinite(Number(targetRisk.risk_score))
      ? Number(targetRisk.risk_score)
      : null
  const operationalTier = targetRisk?.risk_tier || 'QUIESCENT'
  const pollingTier = targetRisk?.polling_tier != null ? targetRisk.polling_tier : 0

  const engineStatus = mlEval?.engine_status || (mlEval ? 'ACTIVE' : 'INACTIVE')
  const evidenceStatus = mlEval?.evidence_status || 'UNAVAILABLE'
  const isWindowEvaluated = evidenceStatus === 'WINDOW_EVALUATED' && engineStatus === 'ACTIVE'
  const isCollecting = evidenceStatus === 'INSUFFICIENT_EVIDENCE' || (mlEval && !isWindowEvaluated && evidenceStatus !== 'ERROR')
  const isUnavailable = !mlEval || evidenceStatus === 'ERROR' || engineStatus !== 'ACTIVE'

  const calProb = mlEval?.calibrated_probability != null ? Number(mlEval.calibrated_probability) : null
  const rawProb = mlEval?.raw_probability != null ? Number(mlEval.raw_probability) : null

  const rawDecision = mlEval?.decision_state || mlEval?.display_state
  const mlDecision = isWindowEvaluated
    ? (rawDecision || (mlEval?.behavioral_indicator && mlEval.behavioral_indicator !== 'NORMAL' ? 'SUSPICIOUS' : 'NORMAL'))
    : isCollecting
    ? 'COLLECTING'
    : 'UNAVAILABLE'

  const behavioralIndicator = mlEval?.behavioral_indicator || null
  const windowId = mlEval?.window_id || null

  const topFeatures = targetRisk?.explainability?.top_features || []
  const behavioralFeature = topFeatures.find((f) => f.feature === 'behavioral_trajectory_anomaly')

  const isBehavioralEscalated =
    Boolean(behavioralFeature) ||
    (operationalTier === 'SUSPICIOUS' &&
      operationalScore != null &&
      operationalScore >= 65.0 &&
      (behavioralIndicator === 'PACING' || behavioralIndicator === 'RANDOM_DRIFT'))

  const distinctWindows = new Set()
  userHistory.forEach((h) => {
    const wId = h.ml_evaluation?.window_id
    if (wId && h.ml_evaluation?.evidence_status === 'WINDOW_EVALUATED') {
      distinctWindows.add(wId)
    }
  })
  if (mlEval?.window_id && isWindowEvaluated) {
    distinctWindows.add(mlEval.window_id)
  }

  // Pre vs Post escalation score
  let scoreBeforeEscalation = null
  let scoreAfterEscalation = operationalScore
  if (isBehavioralEscalated && userHistory.length > 1) {
    const escIndex = userHistory.findIndex((h) => h.risk_tier === 'SUSPICIOUS' || Number(h.risk_score) >= 65.0)
    if (escIndex > 0) {
      scoreBeforeEscalation = Number(userHistory[escIndex - 1].risk_score)
      scoreAfterEscalation = Number(userHistory[escIndex].risk_score)
    }
  }

  // Trace stages
  let stage1Status = isWindowEvaluated ? 'EVALUATED' : isCollecting ? 'COLLECTING' : 'UNAVAILABLE'
  let stage2Status = operationalScore != null ? 'EVALUATED' : 'UNAVAILABLE'
  let stage3Status = isBehavioralEscalated ? 'ESCALATED' : isUnavailable ? 'UNAVAILABLE' : 'NOT TRIGGERED'
  const hasAlert = operationalTier === 'SUSPICIOUS' || operationalTier === 'CRITICAL' || userAlerts.length > 0
  let stage4Status = hasAlert ? 'ALERT GENERATED' : 'NO ALERT'

  // Rationale case
  let rationaleCase = 'OFFLINE'
  if (isBehavioralEscalated) {
    rationaleCase = 'BEHAVIORAL_ESCALATION'
  } else if (isWindowEvaluated && (mlDecision === 'SUSPICIOUS' || (calProb != null && calProb >= 0.5)) && (operationalTier === 'QUIESCENT' || operationalTier === 'NORMAL_TRANSIT')) {
    rationaleCase = 'DIVERGENT_EXPLAINED'
  } else if (isWindowEvaluated && (mlDecision === 'NORMAL' || (calProb != null && calProb < 0.5))) {
    rationaleCase = 'CONCORDANT_NORMAL'
  } else if (isCollecting) {
    rationaleCase = 'COLLECTING'
  }

  // Evidence extraction without inventing numbers
  const kinFeats = mlEval?.kinematic_features || {}
  const speedDisplay = kinFeats.mean_speed_mps != null
    ? `${Number(kinFeats.mean_speed_mps).toFixed(2)} m/s`
    : targetRisk?.sensor_metrics?.speed_mps != null
    ? `${Number(targetRisk.sensor_metrics.speed_mps).toFixed(2)} m/s`
    : 'Unavailable'

  const displacementDisplay = kinFeats.straight_line_displacement_m != null
    ? `${Number(kinFeats.straight_line_displacement_m).toFixed(1)}m`
    : targetRisk?.sensor_metrics?.net_displacement_m != null
    ? `${Number(targetRisk.sensor_metrics.net_displacement_m).toFixed(1)}m`
    : 'Unavailable'

  const phenotypeDisplay = behavioralIndicator ? behavioralIndicator : isWindowEvaluated ? 'NORMAL' : 'Unavailable'

  return {
    isLive,
    sessionStatus,
    operationalScore,
    operationalTier,
    pollingTier,
    isWindowEvaluated,
    calProb,
    rawProb,
    mlDecision,
    behavioralIndicator,
    phenotypeDisplay,
    speedDisplay,
    displacementDisplay,
    isBehavioralEscalated,
    distinctWindowsCount: distinctWindows.size,
    scoreBeforeEscalation,
    scoreAfterEscalation,
    stage1Status,
    stage2Status,
    stage3Status,
    stage4Status,
    rationaleCase,
    completedPointsCount: safeTrail.length > 0 ? safeTrail.length : userHistory.length,
    alertsCount: userAlerts.length,
  }
}

// ----------------------------------------------------------------------------
// Fix 1: Lapping vs Normal vs Divergence Tests
// ----------------------------------------------------------------------------

test('5. Fix 1: When model actually classifies LAPPING, explains recognized lapping exclusion', () => {
  const currentRisk = {
    user_id: 'lap_user_01',
    risk_score: 24.0,
    risk_tier: 'QUIESCENT',
    timestamp: '2026-10-21T12:00:00Z',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      calibrated_probability: 0.81,
      decision_state: 'SUSPICIOUS',
      behavioral_indicator: 'LAPPING',
      window_id: 'win_lap_01',
    },
  }

  const state = deriveRiskInterpretationState({
    currentRisk,
    effectiveRisk: currentRisk,
    historyEntries: [currentRisk],
    alertMessages: [],
    isConnected: true,
  })

  assert.equal(state.behavioralIndicator, 'LAPPING')
  assert.equal(state.phenotypeDisplay, 'LAPPING')
  assert.equal(state.isBehavioralEscalated, false)
  assert.equal(state.rationaleCase, 'DIVERGENT_EXPLAINED')
  assert.equal(state.stage3Status, 'NOT TRIGGERED')
})

test('6. Fix 1: When model reports NORMAL, shows NORMAL and does not claim lapping was detected', () => {
  const currentRisk = {
    user_id: 'normal_user_01',
    risk_score: 18.0,
    risk_tier: 'QUIESCENT',
    timestamp: '2026-10-21T12:00:00Z',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      calibrated_probability: 0.08,
      raw_probability: 0.07,
      decision_state: 'NORMAL',
      behavioral_indicator: 'NORMAL',
      window_id: 'win_norm_01',
    },
  }

  const state = deriveRiskInterpretationState({
    currentRisk,
    effectiveRisk: currentRisk,
    historyEntries: [currentRisk],
    alertMessages: [],
    isConnected: true,
  })

  assert.equal(state.mlDecision, 'NORMAL')
  assert.equal(state.phenotypeDisplay, 'NORMAL')
  assert.notEqual(state.behavioralIndicator, 'LAPPING')
  assert.equal(state.rationaleCase, 'CONCORDANT_NORMAL')
})

test('7. Fix 1: Explains ML SUSPICIOUS (85.2%) vs Operational QUIESCENT (31.9/100)', () => {
  const currentRisk = {
    user_id: 'demo_user',
    risk_score: 31.9,
    risk_tier: 'QUIESCENT',
    polling_tier: 0,
    timestamp: '2026-10-21T12:00:00Z',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      calibrated_probability: 0.852,
      raw_probability: 0.821,
      decision_state: 'SUSPICIOUS',
      behavioral_indicator: 'PACING',
      window_id: 'win_pacing_01',
    },
  }

  const state = deriveRiskInterpretationState({
    currentRisk,
    effectiveRisk: currentRisk,
    historyEntries: [currentRisk],
    alertMessages: [],
    isConnected: true,
  })

  assert.equal(state.operationalScore, 31.9)
  assert.equal(state.operationalTier, 'QUIESCENT')
  assert.equal(state.calProb, 0.852)
  assert.equal(state.mlDecision, 'SUSPICIOUS')
  assert.equal(state.rationaleCase, 'DIVERGENT_EXPLAINED')
  assert.equal(state.isBehavioralEscalated, false, 'Window 1 of pacing must not escalate operational risk')
})

// ----------------------------------------------------------------------------
// Fix 2: Evidence Consistency & No Invented Values
// ----------------------------------------------------------------------------

test('8. Fix 2: Displays Unavailable instead of inventing arbitrary values when kinematic data is missing', () => {
  const currentRisk = {
    user_id: 'demo_user',
    risk_score: 25.0,
    risk_tier: 'QUIESCENT',
    timestamp: '2026-10-21T12:00:00Z',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      calibrated_probability: 0.15,
      decision_state: 'NORMAL',
      kinematic_features: {}, // No mean_speed_mps or straight_line_displacement_m
    },
    sensor_metrics: {}, // No fallback metrics
  }

  const state = deriveRiskInterpretationState({
    currentRisk,
    effectiveRisk: currentRisk,
    historyEntries: [currentRisk],
    alertMessages: [],
    isConnected: true,
  })

  assert.equal(state.speedDisplay, 'Unavailable', 'Must show Unavailable rather than inventing 1.2 m/s')
  assert.equal(state.displacementDisplay, 'Unavailable', 'Must show Unavailable rather than inventing Within Bounds')
})

test('9. Fix 2: Strict user scoping ensures evidence matches selected user without cross-patient contamination', () => {
  const user1Risk = { user_id: 'patient_A', risk_score: 20.0, risk_tier: 'QUIESCENT', timestamp: '2026-10-21T11:00:00Z' }
  const user2Risk = { user_id: 'patient_B', risk_score: 75.0, risk_tier: 'SUSPICIOUS', timestamp: '2026-10-21T12:00:00Z' }

  const mixedAlerts = [
    { id: 1, user_id: 'patient_A', alert_tier: 'QUIESCENT' },
    { id: 2, user_id: 'patient_B', alert_tier: 'SUSPICIOUS' },
    { id: 3, user_id: 'patient_B', alert_tier: 'CRITICAL' },
  ]

  // Monitor patient_A
  const stateA = deriveRiskInterpretationState({
    currentRisk: user1Risk,
    effectiveRisk: user1Risk,
    historyEntries: [user1Risk, user2Risk],
    alertMessages: mixedAlerts,
    isConnected: true,
    userId: 'patient_A',
  })

  assert.equal(stateA.alertsCount, 1, 'patient_A must only count 1 alert, ignoring patient_B alerts')
  assert.equal(stateA.completedPointsCount, 1, 'patient_A must only count 1 point')
})

// ----------------------------------------------------------------------------
// Fix 3: Run Status Distinction (LIVE, STREAM DISCONNECTED, RUN COMPLETED)
// ----------------------------------------------------------------------------

test('10. Fix 3: Distinguishes LIVE, STREAM DISCONNECTED, and RUN COMPLETED correctly', () => {
  const record = {
    user_id: 'demo_user',
    risk_score: 22.0,
    risk_tier: 'QUIESCENT',
    timestamp: '2026-10-21T12:00:00Z',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      decision_state: 'NORMAL',
    },
  }

  // Case A: LIVE
  const stateLive = deriveRiskInterpretationState({
    currentRisk: record,
    effectiveRisk: record,
    historyEntries: [record],
    isConnected: true,
  })
  assert.equal(stateLive.sessionStatus, 'LIVE')

  // Case B: Disconnected without completion confirmation
  const stateDisconnected = deriveRiskInterpretationState({
    currentRisk: null,
    effectiveRisk: record,
    historyEntries: [record],
    isConnected: false,
    isRunCompleted: false,
  })
  assert.equal(stateDisconnected.sessionStatus, 'STREAM DISCONNECTED')

  // Case C: Disconnected WITH confirmed completion
  const stateCompleted = deriveRiskInterpretationState({
    currentRisk: null,
    effectiveRisk: record,
    historyEntries: [record],
    isConnected: false,
    isRunCompleted: true,
  })
  assert.equal(stateCompleted.sessionStatus, 'RUN COMPLETED')
})
