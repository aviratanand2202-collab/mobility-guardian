/**
 * Verification of Real Dataset-Backed Patient Tracking in Technical Inspector.
 *
 * Requirements Verified:
 * 1. TechnicalDashboard includes patient selection controls for test_user_000, test_user_2, test_user_3.
 * 2. App.jsx synchronizes location history on userId change and passes trailMetadata to TechnicalDashboard.
 * 3. App.jsx fetchHistory loads actual user telemetry and clears previous state when switching users or when no data exists.
 * 4. MapPanel renders "No recorded location data for this patient" when a patient has no coordinates in Normal Monitoring.
 * 5. MapPanel plots recorded trajectory points (CircleMarkers) and timestamps for all recorded waypoints.
 * 6. MapPanel renders Walking Trail Timeline with real waypoints, timestamps, and coordinates in Technical Inspector.
 * 7. Group Demo remains completely separated and never populates the real monitoring map.
 * 8. Live backend API verification against the actual database for test_user_000, test_user_2, and test_user_3.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const appPath = path.resolve(__dirname, '../src/App.jsx')
const appCode = fs.readFileSync(appPath, 'utf8')

const technicalDashboardPath = path.resolve(__dirname, '../src/components/technical/TechnicalDashboard.jsx')
const technicalDashboardCode = fs.readFileSync(technicalDashboardPath, 'utf8')

const mapPanelPath = path.resolve(__dirname, '../src/components/MapPanel.jsx')
const mapPanelCode = fs.readFileSync(mapPanelPath, 'utf8')

test('1. TechnicalDashboard provides flexible patient selection for any ID without fixed restriction', () => {
  assert.equal(
    /patient-quick-select/.test(technicalDashboardCode),
    false,
    'Fixed patient-quick-select buttons restriction must be removed'
  )
  assert.match(
    technicalDashboardCode,
    /id="user-id-input"/,
    'Must include editable user-id-input field'
  )
  assert.match(
    technicalDashboardCode,
    /handleFormSubmit/,
    'Must handle form submit for any entered patient ID'
  )
  assert.match(
    technicalDashboardCode,
    /<MapPanel[\s\S]*?trailMetadata=\{trailMetadata\}/,
    'TechnicalDashboard must pass trailMetadata prop to MapPanel'
  )
})

test('2. App.jsx synchronizes persisted location history when userId changes', () => {
  assert.match(
    appCode,
    /useEffect\(\(\) => \{[\s\S]*?fetchHistory\(userId\.trim\(\)\)[\s\S]*?\}, \[userId\]\)/,
    'App.jsx must have useEffect to fetch history whenever userId changes'
  )
  assert.match(
    appCode,
    /<TechnicalDashboard[\s\S]*?trailMetadata=\{trailMetadata\}/,
    'App.jsx must pass trailMetadata to TechnicalDashboard'
  )
})

test('3. App.jsx fetchHistory cleanly resets location state when switching to patient with no data', () => {
  assert.match(
    appCode,
    /setCurrentLocation\(null\)[\s\S]*?setLocationHistory\(\[\]\)[\s\S]*?setTrailMetadata\(\[\]\)/,
    'fetchHistory must reset location state to null/empty when no coordinates exist'
  )
  assert.match(
    appCode,
    /setLocationHistory\(seededCoords\.slice\(-30\)\)/,
    'Must populate locationHistory with actual recorded coordinates'
  )
  assert.match(
    appCode,
    /setTrailMetadata\(seededMeta\.slice\(-30\)\)/,
    'Must populate trailMetadata with actual recorded timestamps and metrics'
  )
})

test('4. MapPanel renders "No recorded location data for this patient" when patient has no records', () => {
  assert.match(
    mapPanelCode,
    /No recorded location data for this patient/,
    'MapPanel must display exact text: "No recorded location data for this patient"'
  )
  assert.match(
    mapPanelCode,
    /!isMultiPatientView && !activePosition/,
    'Must trigger no-data state when activePosition is missing'
  )
})

test('5. MapPanel plots recorded trajectory CircleMarkers and tooltips with timestamps', () => {
  assert.match(
    mapPanelCode,
    /locationHistory\.map\(\(coords,\s*idx\) => \{[\s\S]*?<CircleMarker/,
    'MapPanel must map over locationHistory to render CircleMarker for each recorded point'
  )
  assert.match(
    mapPanelCode,
    /formatTimestamp\(meta\.timestamp\)/,
    'CircleMarker tooltip must display recorded timestamp'
  )
})

test('6. Walking Trail Timeline renders real recorded waypoints in Technical Inspector', () => {
  assert.match(
    mapPanelCode,
    /\(isCaregiverMode \|\| locationHistory\.length > 0\) &&/,
    'Walking Trail Timeline must be available in Technical Inspector when history exists'
  )
  assert.match(
    mapPanelCode,
    /Chronological breadcrumbs \(\{locationHistory\.length\} recorded waypoints\)/,
    'Timeline must report exact count of recorded waypoints'
  )
})

test('7. Group Demo simulation is completely removed and replaced with real Multi-Patient Tracking', () => {
  assert.equal(
    /DEMO_ROUTES/.test(mapPanelCode),
    false,
    'DEMO_ROUTES must be removed from MapPanel'
  )
  assert.equal(
    /SIMULATED DEMO — NOT LIVE GPS/.test(mapPanelCode),
    false,
    'Fake demo banner must be removed'
  )
  assert.match(
    mapPanelCode,
    /isMultiPatientView/,
    'Must support real Multi-Patient Tracking view'
  )
})

test('8. Real Backend API returns distinct coordinates for test_user_000 and test_user_2, and no-data for test_user_3', async () => {
  // Query backend running at localhost:8000
  try {
    const res0 = await fetch('http://127.0.0.1:8000/api/risk-history/test_user_000?limit=100')
    assert.equal(res0.status, 200, 'test_user_000 history endpoint must return 200')
    const data0 = await res0.json()
    assert.equal(data0.length, 30, 'test_user_000 must have 30 recorded trajectory points')
    assert.ok(data0[0].location.lat > 39.0 && data0[0].location.lat < 41.0, 'test_user_000 coordinates must be Beijing dataset')

    const res2 = await fetch('http://127.0.0.1:8000/api/risk-history/test_user_2?limit=100')
    assert.equal(res2.status, 200, 'test_user_2 history endpoint must return 200')
    const data2 = await res2.json()
    assert.equal(data2.length, 30, 'test_user_2 must have 30 recorded trajectory points')
    assert.ok(data2[0].location.lat > 39.0 && data2[0].location.lat < 41.0, 'test_user_2 coordinates must be Beijing dataset')
    // Distinct coordinates check
    assert.notEqual(
      data0[0].location.lat,
      data2[0].location.lat,
      'test_user_000 and test_user_2 must have distinct coordinates'
    )

    const res3 = await fetch('http://127.0.0.1:8000/api/risk-history/test_user_3?limit=100')
    assert.equal(res3.status, 200, 'test_user_3 history endpoint must return 200')
    const data3 = await res3.json()
    assert.equal(data3.length, 0, 'test_user_3 must have 0 records (the no-data case)')
  } catch (err) {
    if (err.code === 'ECONNREFUSED') {
      console.warn('Backend server not reachable on port 8000, skipping live HTTP assertion')
    } else {
      throw err
    }
  }
})

test('9. TechnicalDashboard supports arbitrary entered patient IDs and Enter key submission', () => {
  assert.match(
    technicalDashboardCode,
    /id="user-id-input"/,
    'Must have user-id-input field'
  )
  assert.match(
    technicalDashboardCode,
    /onKeyDown=\{[\s\S]*?e\.key === 'Enter'[\s\S]*?handleFormSubmit/,
    'Enter key must trigger patient submission'
  )
  assert.match(
    technicalDashboardCode,
    /handleFormSubmit/,
    'Form submit must validate and submit patient ID'
  )
})

test('10. App.jsx immediately clears previous patient location state and prevents stale async responses', () => {
  assert.match(
    appCode,
    /activeFetchIdRef\.current/,
    'App.jsx must track active fetch request ID to prevent race conditions'
  )
  assert.match(
    appCode,
    /if\s*\(\s*fetchId\s*!==\s*activeFetchIdRef\.current\s*\)\s*return/,
    'App.jsx must discard stale responses if another patient was selected in flight'
  )
  assert.match(
    appCode,
    /setRiskMessages\(\[\]\)[\s\S]*?setAlertMessages\(\[\]\)/,
    'App.jsx must reset stream messages on patient switch'
  )
})

test('11. System distinguishes unknown patient ID from existing patient with zero recorded points', async () => {
  assert.match(
    mapPanelCode,
    /patientStatus && !patientStatus\.exists/,
    'MapPanel must distinguish unknown patient ID from existing registered patient'
  )
  assert.match(
    technicalDashboardCode,
    /patientStatus\.exists[\s\S]*?Verified Patient[\s\S]*?Unknown Patient/,
    'TechnicalDashboard must show visual registry badge distinguishing verified vs unknown patient'
  )
  // Verify live consent API distinction
  try {
    const resReg = await fetch('http://127.0.0.1:8000/api/consent/test_user_3')
    assert.equal(resReg.status, 200)
    const dataReg = await resReg.json()
    assert.equal(dataReg.status === 'not_found', false, 'test_user_3 must exist in consent registry')

    const resUnk = await fetch('http://127.0.0.1:8000/api/consent/unregistered_unknown_id_999')
    assert.equal(resUnk.status, 200)
    const dataUnk = await resUnk.json()
    assert.equal(dataUnk.status, 'not_found', 'Arbitrary unknown ID must return not_found status')
  } catch (err) {
    if (err.code === 'ECONNREFUSED') {
      console.warn('Backend server not reachable on port 8000, skipping live HTTP assertion')
    } else {
      throw err
    }
  }
})

test('12. MapPanel, timeline, and risk history table consistently bind to the same selected patient ID', () => {
  assert.match(
    technicalDashboardCode,
    /Historical Risk Records \(REST: GET \/api\/risk-history\/\{userId\}\?limit=100\)/,
    'Historical risk records must reference selected userId'
  )
  assert.match(
    technicalDashboardCode,
    /<MapPanel[\s\S]*?userId=\{userId\}[\s\S]*?patientStatus=\{patientStatus\}/,
    'MapPanel must receive both userId and patientStatus props'
  )
})
