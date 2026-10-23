/**
 * Automated regression test suite for:
 * 1. Multi-Patient View Empty State & Patient Selection (Requirement A)
 * 2. Trajectory Segmentation & Session Boundary Isolation (Requirement B)
 * 3. Heartbeat Data Provenance & Freshness (Requirement C)
 * 4. Group Separation & Caregiver Alert Integration (Requirement D)
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

const mapPanelCode = fs.readFileSync(
  path.resolve('frontend/src/components/MapPanel.jsx'),
  'utf8'
)
const appCode = fs.readFileSync(
  path.resolve('frontend/src/App.jsx'),
  'utf8'
)
const caregiverCode = fs.readFileSync(
  path.resolve('frontend/src/components/caregiver/CaregiverDashboard.jsx'),
  'utf8'
)
const riskPanelCode = fs.readFileSync(
  path.resolve('frontend/src/components/RiskPanel.jsx'),
  'utf8'
)

/* ==========================================================================
   Part 1: Multi-Patient View Empty State (Requirement A)
   ========================================================================== */

test('1. Multi-Patient View initializes without fabricated or hardcoded patient IDs', () => {
  // Must NOT initialize with hardcoded ['test_user_000', 'test_user_2']
  assert.equal(
    /multiPatientIds,\s*setMultiPatientIds\]\s*=\s*useState\(\['test_user_000'/.test(mapPanelCode),
    false,
    'Must not initialize multiPatientIds with hardcoded test_user_000'
  )
  assert.match(
    mapPanelCode,
    /multiPatientIds,\s*setMultiPatientIds\]\s*=\s*useState\(\[\]\)/,
    'Must initialize multiPatientIds as an empty array'
  )
})

test('2. Multi-Patient View renders clear empty state when no comparison patients are selected', () => {
  assert.match(
    mapPanelCode,
    /!hasComparisonSelection[\s\S]*?No comparison patient selected/i,
    'Map canvas must display dedicated empty state when no comparison patient is selected'
  )
  assert.match(
    mapPanelCode,
    /Enter a comparison patient ID/i,
    'Empty state must provide clear instructions to add a comparison patient'
  )
  assert.match(
    mapPanelCode,
    /multi-patient-empty-notice[\s\S]*?No Comparison Patient Selected/,
    'Summary timeline section must display empty notice without rendering fabricated cards'
  )
})

test('3. Multi-Patient View distinguishes loading, empty (0 pts), error, and unknown patient states', () => {
  assert.match(
    mapPanelCode,
    /Unknown ID|Unknown Patient ID/i,
    'Must distinguish unknown patient ID from existing patients'
  )
  assert.match(
    mapPanelCode,
    /0 points|0 pts/i,
    'Must clearly indicate registered patient with zero recorded points'
  )
  assert.match(
    mapPanelCode,
    /Loading\.\.\./,
    'Must indicate loading state while records are being fetched'
  )
})

/* ==========================================================================
   Part 2: Trajectory Segmentation & Session Boundary (Requirement B)
   ========================================================================== */

test('4. MapPanel defines splitIntoContinuousSegments to prevent cross-continent straight lines', () => {
  assert.match(
    mapPanelCode,
    /function splitIntoContinuousSegments\(coords,\s*metadata/,
    'Must define splitIntoContinuousSegments helper'
  )
  assert.match(
    mapPanelCode,
    /trajectorySegments\.map\(\(segment,\s*segIdx\)/,
    'Must map over trajectorySegments to render independent Polyline segments'
  )
})

test('5. App.jsx isolates latest contiguous session and prevents distant historical jumps', () => {
  assert.match(
    appCode,
    /dLat > 0\.45 \|\| dLng > 0\.45 \|\| timeDiffMin > 30/,
    'App.jsx must identify session boundaries when distance jump > 50km or time gap > 30 min'
  )
  assert.match(
    appCode,
    /if \(dLat > 0\.45 \|\| dLng > 0\.45\) \{\s*return \[\[loc\.lat,\s*loc\.lng\]\]/,
    'Live WebSocket ingest must start fresh session segment on massive geographic jump'
  )
})

test('6. Bounds fitting targets active session rather than spanning all historical points', () => {
  assert.match(
    mapPanelCode,
    /activeSessionBounds[\s\S]*?splitIntoContinuousSegments/,
    'Must derive activeSessionBounds from the latest contiguous segment'
  )
})

/* ==========================================================================
   Part 3: Heartbeat Provenance (Requirement C)
   ========================================================================== */

test('7. CaregiverDashboard clearly displays heartbeat data provenance (Simulated vs Hardware)', () => {
  assert.match(
    caregiverCode,
    /isSimulated\s*=\s*sensorStatus === 'SIMULATED_TEST_STREAM'/,
    'Must identify simulated test stream from sensor_status'
  )
  assert.match(
    caregiverCode,
    /\(Simulated\)|\(Live\)/,
    'Must suffix displayed BPM with provenance label'
  )
  assert.match(
    caregiverCode,
    /metric-header-row[\s\S]*?Simulated[\s\S]*?Hardware/,
    'Must render visual provenance badge in heartbeat metric card'
  )
})

test('8. RiskPanel renders data provenance details and explicit unavailable state', () => {
  assert.match(
    riskPanelCode,
    /Data Provenance:/,
    'Must display Data Provenance label in biometrics card'
  )
  assert.match(
    riskPanelCode,
    /provenanceLabel/,
    'Must render formatted provenance label in biometrics inspector'
  )
  assert.match(
    riskPanelCode,
    /No wearable telemetry received/,
    'Must indicate explicit unavailable state when heart rate is absent'
  )
})

/* ==========================================================================
   Part 4: Group Separation & Companion Integration (Requirement D)
   ========================================================================== */

test('9. MapPanel renders companion accompaniment status and visual separation line', () => {
  assert.match(
    mapPanelCode,
    /activeCompanion\.accompaniment_status === 'Separated'/,
    'MapPanel must distinguish Separated vs With Companion visually'
  )
  assert.match(
    mapPanelCode,
    /activeCompanion\.accompaniment_status \|\| 'Accompaniment Unverified'/,
    'MapPanel must display accompaniment status accurately'
  )
})
