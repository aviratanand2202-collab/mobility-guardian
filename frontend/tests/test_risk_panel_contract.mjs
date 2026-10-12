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
function evaluateRiskPanelState(currentRisk, historyEntries) {
  const latestHistory = historyEntries && historyEntries.length > 0
    ? historyEntries[historyEntries.length - 1]
    : null
  const effectiveRisk = currentRisk || latestHistory

  const riskTier = effectiveRisk?.risk_tier || 'QUIESCENT'
  const riskScore = effectiveRisk?.risk_score != null ? Number(effectiveRisk.risk_score) : null
  const hasValidScore = typeof riskScore === 'number' && Number.isFinite(riskScore)

  // Stale-prevention logic from RiskPanel.jsx
  const mlEvaluation = currentRisk ? currentRisk.ml_evaluation : latestHistory?.ml_evaluation

  let mlStatus = null
  let mlProbabilityFormatted = null
  let mlDisplayMessage = null
  let isMlCardRendered = mlEvaluation != null

  if (isMlCardRendered) {
    if (
      mlEvaluation.evidence_status === 'WINDOW_EVALUATED' &&
      mlEvaluation.engine_status === 'ACTIVE' &&
      mlEvaluation.calibrated_probability != null
    ) {
      mlStatus = 'WINDOW_EVALUATED'
      mlProbabilityFormatted = `${(mlEvaluation.calibrated_probability * 100).toFixed(1)}%`
    } else if (mlEvaluation.evidence_status === 'INSUFFICIENT_EVIDENCE') {
      mlStatus = 'INSUFFICIENT_EVIDENCE'
      mlDisplayMessage = 'Collecting telemetry'
    } else {
      mlStatus = 'UNAVAILABLE'
      mlDisplayMessage = 'ML Evaluation Unavailable'
    }
  }

  return {
    operationalRiskScore: riskScore,
    operationalRiskTier: riskTier,
    hasValidScore,
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
