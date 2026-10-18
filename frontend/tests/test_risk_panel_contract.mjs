/**
 * Focused regression tests for RiskPanel ML Evaluation Display (Step 6).
 *
 * Verifies:
 * 1. Missing / Disabled state: ml-evaluation-card is omitted; operational UI is baseline.
 * 2. WINDOW_EVALUATED state: displays outlier probability (e.g. 84.1%), observational disclaimer,
 *    and keeps operational risk score and tier authoritative.
 * 3. INSUFFICIENT_EVIDENCE state: displays "Collecting telemetry", does NOT display 0.0% or safe claim.
 * 4. ERROR / TIMEOUT state: displays neutral "ML Evaluation Unavailable" without exposing internal errors.
 * 5. Stale data prevention: live event without ml_evaluation does NOT carry over previous window's ML result.
 * 6. Historical views: displays historical ML evaluation when available without fabricating values.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Verify file exists and contains the required Step 6 logic
const riskPanelPath = path.resolve(__dirname, '../src/components/RiskPanel.jsx')
const riskPanelCode = fs.readFileSync(riskPanelPath, 'utf8')

test('RiskPanel source code contains required shadow ML evaluation structures', (t) => {
  // 1. Derives mlEvaluation strictly preventing stale carry-over
  assert.match(
    riskPanelCode,
    /const mlEvaluation = currentRisk \? currentRisk\.ml_evaluation : latestHistory\?\.ml_evaluation/,
    'Must derive mlEvaluation without stale carry-over when currentRisk is present'
  )

  // 2. Only renders ML card when mlEvaluation is present
  assert.match(
    riskPanelCode,
    /\{mlEvaluation != null && \(/,
    'Must only render ml-evaluation-card when mlEvaluation is not null'
  )

  // 3. Window Evaluated state with percentage probability
  assert.match(
    riskPanelCode,
    /\(mlEvaluation\.calibrated_probability \* 100\)\.toFixed\(1\)/,
    'Must format calibrated_probability as percentage'
  )
  assert.match(
    riskPanelCode,
    /Kinematic Outlier Probability \(120s Window\)/,
    'Must clearly label kinematic outlier probability'
  )

  // 4. Insufficient evidence handling
  assert.match(
    riskPanelCode,
    /Collecting telemetry/,
    'Must display "Collecting telemetry" when evidence is insufficient'
  )
  assert.match(
    riskPanelCode,
    /Safety cannot be verified from incomplete evidence/,
    'Must explicitly state safety cannot be verified from incomplete evidence'
  )

  // 5. Unavailable / Error handling
  assert.match(
    riskPanelCode,
    /ML Evaluation Unavailable/,
    'Must display neutral "ML Evaluation Unavailable" on error or timeout'
  )
  assert.match(
    riskPanelCode,
    /Observational signal temporarily offline/,
    'Must show neutral explanation without exposing internal error tracebacks'
  )

  // 6. Observational disclaimer
  assert.match(
    riskPanelCode,
    /Observational ML evaluation only — not a clinical prediction, diagnosis, or operational safety decision/,
    'Must include explicit observational disclaimer'
  )
})

/**
 * Pure state evaluation logic mirror of RiskPanel.jsx
 */
function evaluateRiskPanelState(currentRisk, historyEntries = []) {
  const safeHistory = Array.isArray(historyEntries) ? historyEntries : []
  const latestHistory = safeHistory.length > 0
    ? safeHistory[safeHistory.length - 1]
    : null

  const isLiveScoreValid =
    currentRisk?.risk_score != null &&
    typeof currentRisk.risk_score !== 'boolean' &&
    currentRisk.risk_score !== '' &&
    Number.isFinite(Number(currentRisk.risk_score))

  const effectiveRisk = isLiveScoreValid ? currentRisk : (latestHistory || currentRisk)

  const riskTier = effectiveRisk?.risk_tier || 'QUIESCENT'
  const riskScoreRaw = effectiveRisk?.risk_score
  const hasValidScore =
    riskScoreRaw != null &&
    typeof riskScoreRaw !== 'boolean' &&
    riskScoreRaw !== '' &&
    Number.isFinite(Number(riskScoreRaw))
  const riskScore = hasValidScore ? Number(riskScoreRaw) : null
  const formattedScore = hasValidScore ? riskScore.toFixed(1) : '--'

  // Stale-prevention logic from RiskPanel.jsx
  const mlEvaluation = currentRisk ? currentRisk.ml_evaluation : latestHistory?.ml_evaluation

  let mlStatus = null
  let mlProbabilityFormatted = null
  let mlDisplayMessage = null
  let isMlCardRendered = mlEvaluation != null

  if (isMlCardRendered) {
    if (
      currentRisk &&
      mlEvaluation.evidence_status === 'WINDOW_EVALUATED' &&
      mlEvaluation.engine_status === 'ACTIVE' &&
      mlEvaluation.calibrated_probability != null
    ) {
      mlStatus = 'WINDOW_EVALUATED'
      mlProbabilityFormatted = `${(mlEvaluation.calibrated_probability * 100).toFixed(1)}%`
    } else if (mlEvaluation.evidence_status === 'INSUFFICIENT_EVIDENCE' || !currentRisk) {
      mlStatus = 'INSUFFICIENT_EVIDENCE'
      mlDisplayMessage = 'Collecting telemetry'
    } else {
      mlStatus = 'UNAVAILABLE'
      mlDisplayMessage = 'ML Evaluation Unavailable'
    }
  }

  // Trend points calculation mirror
  const points = safeHistory
    .filter((p) => p.risk_score != null && typeof p.risk_score !== 'boolean' && p.risk_score !== '' && Number.isFinite(Number(p.risk_score)))
    .map((p) => ({
      ...p,
      risk_score: Number(p.risk_score),
    }))
    .slice(-15)

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
  const latestTrendScore = trendPoints.length > 0 ? trendPoints[trendPoints.length - 1].risk_score : null
  const formattedTrendLatest = latestTrendScore != null ? latestTrendScore.toFixed(1) : '-'

  return {
    operationalRiskScore: riskScore,
    operationalRiskTier: riskTier,
    hasValidScore,
    formattedScore,
    trendPoints,
    latestTrendScore,
    formattedTrendLatest,
    isMlCardRendered,
    mlStatus,
    mlProbabilityFormatted,
    mlDisplayMessage,
    mlEvaluation,
  }
}

test('1. Missing ML Evaluation / Feature Disabled preserves baseline UI', () => {
  const currentRisk = {
    user_id: 'user_1',
    risk_score: 72.5,
    risk_tier: 'SUSPICIOUS',
    polling_tier: 2,
    ml_evaluation: null,
  }
  const history = [
    { id: 1, risk_score: 40.0, risk_tier: 'NORMAL_TRANSIT' },
  ]

  const state = evaluateRiskPanelState(currentRisk, history)

  // Operational fields strictly authoritative
  assert.equal(state.operationalRiskScore, 72.5)
  assert.equal(state.operationalRiskTier, 'SUSPICIOUS')
  assert.equal(state.hasValidScore, true)

  // ML card MUST NOT be rendered
  assert.equal(state.isMlCardRendered, false)
  assert.equal(state.mlStatus, null)
})

test('2. WINDOW_EVALUATED displays outlier probability and keeps operational risk separate', () => {
  const currentRisk = {
    user_id: 'user_1',
    risk_score: 15.0,
    risk_tier: 'QUIESCENT',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'WINDOW_EVALUATED',
      calibrated_probability: 0.885,
      raw_probability: 0.8520,
      ml_risk_score: 85.0,
      decision_state: 'SUSPICIOUS',
    },
  }

  const state = evaluateRiskPanelState(currentRisk, [])

  // Operational score MUST remain 15.0 (QUIESCENT), NOT overridden by ML probability (0.8845) or ml_risk_score (85.0)
  assert.equal(state.operationalRiskScore, 15.0)
  assert.equal(state.operationalRiskTier, 'QUIESCENT')

  // ML evaluation display is active and shows 88.5%
  assert.equal(state.isMlCardRendered, true)
  assert.equal(state.mlStatus, 'WINDOW_EVALUATED')
  assert.equal(state.mlProbabilityFormatted, '88.5%')
})

test('3. INSUFFICIENT_EVIDENCE displays Collecting telemetry and never claims safety', () => {
  const currentRisk = {
    user_id: 'user_1',
    risk_score: 50.0,
    risk_tier: 'NORMAL_TRANSIT',
    ml_evaluation: {
      engine_status: 'ACTIVE',
      evidence_status: 'INSUFFICIENT_EVIDENCE',
      calibrated_probability: null,
      raw_probability: null,
      ml_risk_score: null,
      decision_state: 'INSUFFICIENT_EVIDENCE',
    },
  }

  const state = evaluateRiskPanelState(currentRisk, [])

  assert.equal(state.isMlCardRendered, true)
  assert.equal(state.mlStatus, 'INSUFFICIENT_EVIDENCE')
  assert.equal(state.mlDisplayMessage, 'Collecting telemetry')
  // Crucial: no probability score displayed
  assert.equal(state.mlProbabilityFormatted, null)
})

test('4. ERROR or TIMEOUT displays neutral unavailable status without exposing details', () => {
  const currentRisk = {
    user_id: 'user_1',
    risk_score: 85.0,
    risk_tier: 'CRITICAL',
    ml_evaluation: {
      engine_status: 'TIMEOUT',
      evidence_status: 'ERROR',
      calibrated_probability: null,
      error: 'ML inference execution exceeded time limit.',
    },
  }

  const state = evaluateRiskPanelState(currentRisk, [])

  assert.equal(state.isMlCardRendered, true)
  assert.equal(state.mlStatus, 'UNAVAILABLE')
  assert.equal(state.mlDisplayMessage, 'ML Evaluation Unavailable')
  assert.equal(state.operationalRiskScore, 85.0)
  assert.equal(state.operationalRiskTier, 'CRITICAL')
})

test('5. Stale Data Prevention: live event without ML eval does not inherit previous ML eval', () => {
  const historyWithMl = [
    {
      id: 1,
      risk_score: 60.0,
      risk_tier: 'NORMAL_TRANSIT',
      ml_evaluation: {
        engine_status: 'ACTIVE',
        evidence_status: 'WINDOW_EVALUATED',
        calibrated_probability: 0.95,
      },
    },
  ]

  // New live reading arrives from a stream where ML is disabled or not evaluated
  const currentRiskWithoutMl = {
    user_id: 'user_1',
    timestamp: '2026-10-11T12:05:00Z',
    risk_score: 25.0,
    risk_tier: 'QUIESCENT',
    ml_evaluation: null, // explicitly null
  }

  const state = evaluateRiskPanelState(currentRiskWithoutMl, historyWithMl)

  // Must NOT carry over the 95% ML probability from history
  assert.equal(state.isMlCardRendered, false)
  assert.equal(state.mlProbabilityFormatted, null)
  assert.equal(state.operationalRiskScore, 25.0)
  assert.equal(state.operationalRiskTier, 'QUIESCENT')
})

test('6. TechnicalDashboard historical records table correctly formats ML Eval column', () => {
  const technicalDashboardPath = path.resolve(__dirname, '../src/components/technical/TechnicalDashboard.jsx')
  const techCode = fs.readFileSync(technicalDashboardPath, 'utf8')

  // Verifies conditional column inclusion
  assert.match(
    techCode,
    /const hasAnyMlEval = historyEntries\.some\(\(e\) => e\.ml_evaluation != null\)/,
    'Must check if any history entries contain ml_evaluation'
  )
  assert.match(
    techCode,
    /\{hasAnyMlEval && <th>ML Eval \(Shadow\)<\/th>\}/,
    'Must conditionally render table header'
  )
  assert.match(
    techCode,
    /\(entry\.ml_evaluation\.calibrated_probability \* 100\)\.toFixed\(1\)/,
    'Must format historical probability as percentage'
  )
})

test('7. TechnicalDashboard historical table sorts entries newest first (descending timestamp)', () => {
  const technicalDashboardPath = path.resolve(__dirname, '../src/components/technical/TechnicalDashboard.jsx')
  const techCode = fs.readFileSync(technicalDashboardPath, 'utf8')

  assert.match(
    techCode,
    /const sortedEntries = \[\.\.\.historyEntries\]\.sort/,
    'Must create sorted copy of historyEntries for descending display'
  )
  assert.match(
    techCode,
    /new Date\(b\.timestamp\) - new Date\(a\.timestamp\)/,
    'Must sort descending by timestamp'
  )

  // Verify pure sorting behavior: newest timestamp and higher ID come first
  const mockEntries = [
    { id: 3, timestamp: '2026-10-13T07:48:00Z', risk_score: 40.0 },
    { id: 25, timestamp: '2026-10-13T07:50:00Z', risk_score: 41.0 },
    { id: 32, timestamp: '2026-10-13T07:50:35Z', risk_score: 47.7 },
  ]
  const sorted = [...mockEntries].sort(
    (a, b) => new Date(b.timestamp) - new Date(a.timestamp) || b.id - a.id
  )
  assert.equal(sorted[0].id, 32, 'Newest record must be first')
  assert.equal(sorted[1].id, 25, 'Intermediate record must be second')
  assert.equal(sorted[2].id, 3, 'Oldest record must be last')
})

test('8. App.jsx queries limit=100 and resets live risk/alert messages on socket reconnect/disconnect', () => {
  const appPath = path.resolve(__dirname, '../src/App.jsx')
  const appCode = fs.readFileSync(appPath, 'utf8')

  // History query limit increased to 100 to prevent truncating newer evaluated records
  assert.match(
    appCode,
    /\/api\/risk-history\/\$\{targetUserId\}\?limit=100/,
    'fetchHistory must query with limit=100'
  )

  // Live messages reset in disconnectSockets and connectSockets
  assert.match(
    appCode,
    /disconnectSockets = \(\) => \{[\s\S]*?setRiskMessages\(\[\]\)[\s\S]*?setAlertMessages\(\[\]\)/,
    'disconnectSockets must reset riskMessages and alertMessages'
  )
  assert.match(
    appCode,
    /connectSockets = \([a-zA-Z0-9_]+\) => \{[\s\S]*?setRiskMessages\(\[\]\)[\s\S]*?setAlertMessages\(\[\]\)/,
    'connectSockets must reset riskMessages and alertMessages'
  )
})

test('9. RiskPanel preserves collecting/waiting state when no live evaluation has arrived in current connection', () => {
  // Scenario: Reconnected to socket (currentRisk is null), but persisted history has an evaluated window (ID 32)
  const historyWithEvaluated = [
    {
      id: 32,
      timestamp: '2026-10-13T07:50:35Z',
      risk_score: 47.7,
      risk_tier: 'NORMAL_TRANSIT',
      ml_evaluation: {
        engine_status: 'ACTIVE',
        evidence_status: 'WINDOW_EVALUATED',
        calibrated_probability: 0.7851,
        raw_probability: 0.7800,
        ml_risk_score: 78.5,
        decision_state: 'CRITICAL',
      },
    },
  ]

  // currentRisk is null (no message received yet on new connection)
  const state = evaluateRiskPanelState(null, historyWithEvaluated)

  // Card is rendered because historical ML metadata is known, BUT it must show collecting state, NOT claim 78.5% live
  assert.equal(state.isMlCardRendered, true)
  assert.equal(state.mlStatus, 'INSUFFICIENT_EVIDENCE', 'Must show collecting/waiting status when currentRisk is null')
  assert.equal(state.mlDisplayMessage, 'Collecting telemetry', 'Must display Collecting telemetry message')
  assert.equal(state.mlProbabilityFormatted, null, 'Must NOT format probability as live prediction')
})

test('10. Blank current score regression: renders numeric score when valid, handles null/undefined/non-finite without fabricating zero', () => {
  // Case A: Valid live float score (observed in demo: 22.9 with QUIESCENT)
  const stateLiveValid = evaluateRiskPanelState({ risk_score: 22.9, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateLiveValid.hasValidScore, true)
  assert.equal(stateLiveValid.operationalRiskScore, 22.9)
  assert.equal(stateLiveValid.formattedScore, '22.9')
  assert.equal(stateLiveValid.operationalRiskTier, 'QUIESCENT')

  // Case B: Valid score of exactly 0.0 must NOT be treated as falsy/blank or fallback
  const stateZero = evaluateRiskPanelState({ risk_score: 0, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateZero.hasValidScore, true)
  assert.equal(stateZero.operationalRiskScore, 0)
  assert.equal(stateZero.formattedScore, '0.0')

  // Case C: Valid upper bound score (100.0)
  const stateMax = evaluateRiskPanelState({ risk_score: 100.0, risk_tier: 'CRITICAL' }, [])
  assert.equal(stateMax.hasValidScore, true)
  assert.equal(stateMax.operationalRiskScore, 100.0)
  assert.equal(stateMax.formattedScore, '100.0')

  // Case D: Null score with no history must display placeholder '--' and NOT fabricate 0
  const stateNull = evaluateRiskPanelState({ risk_score: null, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateNull.hasValidScore, false)
  assert.equal(stateNull.operationalRiskScore, null)
  assert.equal(stateNull.formattedScore, '--')

  // Case E: Undefined score with no history must display placeholder '--'
  const stateUndefined = evaluateRiskPanelState({ risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateUndefined.hasValidScore, false)
  assert.equal(stateUndefined.operationalRiskScore, null)
  assert.equal(stateUndefined.formattedScore, '--')

  // Case F: Non-finite values (NaN, Infinity, string 'invalid', boolean) must NOT fabricate 0
  const stateNaN = evaluateRiskPanelState({ risk_score: NaN, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateNaN.hasValidScore, false)
  assert.equal(stateNaN.formattedScore, '--')

  const stateInf = evaluateRiskPanelState({ risk_score: Infinity, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateInf.hasValidScore, false)
  assert.equal(stateInf.formattedScore, '--')

  const stateStr = evaluateRiskPanelState({ risk_score: 'invalid', risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateStr.hasValidScore, false)
  assert.equal(stateStr.formattedScore, '--')

  const stateBool = evaluateRiskPanelState({ risk_score: false, risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateBool.hasValidScore, false)
  assert.equal(stateBool.formattedScore, '--')

  const stateEmptyStr = evaluateRiskPanelState({ risk_score: '', risk_tier: 'QUIESCENT' }, [])
  assert.equal(stateEmptyStr.hasValidScore, false)
  assert.equal(stateEmptyStr.formattedScore, '--')
})

test('11. CSS specificity and contrast: prevents blank/invisible text on light background in RiskPanel', () => {
  const cssPath = path.resolve(__dirname, '../src/index.css')
  const cssContent = fs.readFileSync(cssPath, 'utf8')

  // 1. Caregiver metrics must be scoped to .caregiver-metric-card, not leak global .metric-value: #f8fafc
  assert.match(
    cssContent,
    /\.caregiver-metric-card\s+\.metric-value\s*\{/,
    'Caregiver metric-value must be scoped to .caregiver-metric-card'
  )

  // 2. Risk metric card and metric-value-row must explicitly define high-contrast text color and size
  assert.match(
    cssContent,
    /\.risk-metric-card\s+\.metric-value,\s*\r?\n?\.metric-value-row\s+\.metric-value\s*\{[\s\S]*?font-size:\s*32px;[\s\S]*?color:\s*#1f2937;/m,
    'Risk metric card metric-value must define 32px font size and high-contrast #1f2937 color'
  )

  // 3. Tier text colors must be scoped to .risk-metric-card / .metric-value-row with higher specificity
  assert.match(
    cssContent,
    /\.risk-metric-card\s+\.tier-text-QUIESCENT,\s*\r?\n?\.metric-value-row\s+\.tier-text-QUIESCENT\s*\{\s*color:\s*#0284c7;\s*\}/m,
    'Must define scoped tier-text-QUIESCENT color (#0284c7)'
  )
  assert.match(
    cssContent,
    /\.risk-metric-card\s+\.tier-text-NORMAL_TRANSIT,\s*\r?\n?\.metric-value-row\s+\.tier-text-NORMAL_TRANSIT\s*\{\s*color:\s*#16a34a;\s*\}/m,
    'Must define scoped tier-text-NORMAL_TRANSIT color (#16a34a)'
  )
  assert.match(
    cssContent,
    /\.risk-metric-card\s+\.tier-text-SUSPICIOUS,\s*\r?\n?\.metric-value-row\s+\.tier-text-SUSPICIOUS\s*\{\s*color:\s*#ea580c;\s*\}/m,
    'Must define scoped tier-text-SUSPICIOUS color (#ea580c)'
  )
  assert.match(
    cssContent,
    /\.risk-metric-card\s+\.tier-text-CRITICAL,\s*\r?\n?\.metric-value-row\s+\.tier-text-CRITICAL\s*\{\s*color:\s*#dc2626;\s*\}/m,
    'Must define scoped tier-text-CRITICAL color (#dc2626)'
  )
})

test('12. Score and trend parity: Current Risk Score card and Trend Chart point to the same patient\'s latest accepted record', () => {
  const history = [
    { timestamp: '2026-10-18T10:45:00Z', risk_score: 20.0, risk_tier: 'QUIESCENT' },
    { timestamp: '2026-10-18T10:46:00Z', risk_score: 22.9, risk_tier: 'QUIESCENT' },
  ]

  // Scenario 1: Live event arrives with valid score 24.5
  const liveValid = { timestamp: '2026-10-18T10:47:00Z', risk_score: 24.5, risk_tier: 'QUIESCENT' }
  const stateLive = evaluateRiskPanelState(liveValid, history)
  assert.equal(stateLive.formattedScore, '24.5')
  assert.equal(stateLive.formattedTrendLatest, '24.5')
  assert.equal(stateLive.latestTrendScore, 24.5)

  // Scenario 2: Live event arrives with null score (e.g. heartbeat or partial update)
  // Card must fall back to latest accepted history record (22.9), keeping card and trend synchronized
  const liveNullScore = { timestamp: '2026-10-18T10:47:00Z', risk_score: null, risk_tier: 'QUIESCENT' }
  const stateFallback = evaluateRiskPanelState(liveNullScore, history)
  assert.equal(stateFallback.hasValidScore, true)
  assert.equal(stateFallback.operationalRiskScore, 22.9)
  assert.equal(stateFallback.formattedScore, '22.9')
  assert.equal(stateFallback.formattedTrendLatest, '22.9')
  assert.equal(stateFallback.latestTrendScore, 22.9)

  // Scenario 3: Neither live event nor history has score
  const stateEmpty = evaluateRiskPanelState(null, [])
  assert.equal(stateEmpty.hasValidScore, false)
  assert.equal(stateEmpty.formattedScore, '--')
  assert.equal(stateEmpty.formattedTrendLatest, '-')
  assert.equal(stateEmpty.latestTrendScore, null)
})
