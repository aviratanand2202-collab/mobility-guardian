import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

const DASHBOARD_PATH = path.resolve('frontend/src/components/modelComparison/ModelComparisonDashboard.jsx')
const EVAL_RESULTS_PATH = path.resolve('ml/reports/false_positive_reduction/evaluation_results.json')
const SELECTED_THRESHOLDS_PATH = path.resolve('ml/reports/false_positive_reduction/selected_thresholds.json')

test('1. ModelComparisonDashboard source code exists and defines expected props and state', () => {
  assert.ok(fs.existsSync(DASHBOARD_PATH), 'ModelComparisonDashboard.jsx must exist')
  const src = fs.readFileSync(DASHBOARD_PATH, 'utf-8')
  assert.ok(src.includes('selectedHorizon'), 'Component must maintain selectedHorizon state')
  assert.ok(src.includes('viewVariant'), 'Component must maintain viewVariant state')
  assert.ok(src.includes('fetchComparisonData'), 'Component must fetch comparison data from API')
  assert.ok(src.includes('/api/models/comparison'), 'Component must call /api/models/comparison endpoint')
})

test('2. Terminology: Misleading clinical labels are strictly replaced with non-clinical mathematical descriptions', () => {
  const src = fs.readFileSync(DASHBOARD_PATH, 'utf-8')

  // Prohibited misleading clinical wording
  assert.equal(src.includes('Safe Correctly Muted'), false, 'Misleading "Safe Correctly Muted" must be removed')
  assert.equal(src.includes('Safe Triggered Alarm'), false, 'Misleading "Safe Triggered Alarm" must be removed')
  assert.equal(src.includes('Outlier Missed'), false, 'Misleading "Outlier Missed" must be removed')
  assert.equal(src.includes('Outlier Caught'), false, 'Misleading "Outlier Caught" must be removed')

  // Required exact non-clinical mathematical terminology
  assert.ok(
    src.includes('Routine-Like Window Correctly Classified as Negative'),
    'True Negative must be described as "Routine-Like Window Correctly Classified as Negative"'
  )
  assert.ok(
    src.includes('Routine-Like Window Classified as Positive (False Alert)'),
    'False Positive must be described as "Routine-Like Window Classified as Positive (False Alert)"'
  )
  assert.ok(
    src.includes('Kinematic Outlier Classified as Negative'),
    'False Negative must be described as "Kinematic Outlier Classified as Negative"'
  )
  assert.ok(
    src.includes('Kinematic Outlier Correctly Classified as Positive'),
    'True Positive must be described as "Kinematic Outlier Correctly Classified as Positive"'
  )
})

test('3. Scientific Disclaimer and Operational Parameters are preserved', () => {
  const src = fs.readFileSync(DASHBOARD_PATH, 'utf-8')
  assert.ok(src.includes('Scientific Evaluation Parameters & Operational Disclaimer'), 'Disclaimer section must be present')
  assert.ok(src.includes('dataset_info?.disclaimer'), 'Disclaimer text must be bound to dataset_info.disclaimer')
  assert.ok(src.includes('dataset_info?.target_definition'), 'Target definition must be bound to dataset_info.target_definition')
})

test('4. Evaluation artifacts contain exact mathematical calculations for all 12 combinations', () => {
  assert.ok(fs.existsSync(EVAL_RESULTS_PATH), 'evaluation_results.json must exist')
  const evalData = JSON.parse(fs.readFileSync(EVAL_RESULTS_PATH, 'utf-8'))
  const testEvals = evalData.test_evaluations

  const models = ['xgboost_improved', 'random_forest', 'lightgbm']
  const horizons = ['120s', '360s', '600s', '840s']

  for (const m of models) {
    for (const h of horizons) {
      const hData = testEvals[m][h]
      assert.ok(hData, `Missing evaluation data for ${m} ${h}`)

      const base = hData.baseline
      const rev = hData.revised
      const delta = hData.delta

      // Check baseline confusion matrix
      const cmB = base.confusion_matrix
      const expectedBaseFpr = cmB.fp / (cmB.fp + cmB.tn)
      assert.ok(Math.abs(base.false_positive_rate - expectedBaseFpr) < 1e-4)

      // Check revised confusion matrix
      const cmR = rev.confusion_matrix
      const expectedRevFpr = cmR.fp / (cmR.fp + cmR.tn)
      assert.ok(Math.abs(rev.false_positive_rate - expectedRevFpr) < 1e-4)

      // Check deltas
      assert.equal(delta.delta_fp, cmR.fp - cmB.fp, `delta_fp must equal cmR.fp - cmB.fp for ${m} ${h}`)
      assert.equal(delta.delta_fn, cmR.fn - cmB.fn, `delta_fn must equal cmR.fn - cmB.fn for ${m} ${h}`)
      assert.ok(delta.delta_fp < 0, `Revised threshold must reduce false positives for ${m} ${h}`)
    }
  }
})

test('5. Selected threshold values come from validation metadata and are frozen before test scoring', () => {
  assert.ok(fs.existsSync(SELECTED_THRESHOLDS_PATH), 'selected_thresholds.json must exist')
  const threshData = JSON.parse(fs.readFileSync(SELECTED_THRESHOLDS_PATH, 'utf-8'))
  const evalData = JSON.parse(fs.readFileSync(EVAL_RESULTS_PATH, 'utf-8'))

  const models = ['xgboost_improved', 'random_forest', 'lightgbm']
  const horizons = ['120s', '360s', '600s', '840s']

  for (const m of models) {
    for (const h of horizons) {
      const tunedThreshold = threshData.models[m][h].revised_threshold
      const testRevThreshold = evalData.test_evaluations[m][h].revised.threshold
      assert.ok(
        Math.abs(testRevThreshold - tunedThreshold) < 1e-4,
        `Test evaluation revised threshold must match validation tuned threshold for ${m} ${h}`
      )
    }
  }
})

test('6. App.jsx renders Model Comparison View toggle button and component branch', () => {
  const appSrc = fs.readFileSync(path.resolve('frontend/src/App.jsx'), 'utf-8')
  assert.ok(appSrc.includes("import ModelComparisonDashboard from './components/modelComparison/ModelComparisonDashboard'"))
  assert.ok(appSrc.includes("activeView === 'comparison'"))
  assert.ok(appSrc.includes('<ModelComparisonDashboard />'))
})
