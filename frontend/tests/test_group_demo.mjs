/**
 * Contract and Regression Verification:
 * 1. Complete removal of simulated Group Demo (SF coordinates, fake waypoints, demo banner, simulated users).
 * 2. Real Multi-Patient Tracking view with independent API fetching, distinct patient markers, and automatic bounds fitting.
 * 3. Spatial math calculations (Haversine, Geofence) preserved for spatial verification.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  haversineMeters,
  isInsideGeofence,
  DEFAULT_GEOFENCE,
} from '../src/components/companion/companionLogic.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const mapPanelPath = path.resolve(__dirname, '../src/components/MapPanel.jsx')
const mapPanelCode = fs.readFileSync(mapPanelPath, 'utf8')

const cssPath = path.resolve(__dirname, '../src/index.css')
const cssCode = fs.readFileSync(cssPath, 'utf8')

/* ==========================================================================
   Part 1: Complete Removal of Hardcoded Simulated Group Demo
   ========================================================================== */

test('1. MapPanel completely removes simulated Group Demo routes, users, and imports', () => {
  // Confirm simulation routes and users are NOT imported or defined in MapPanel
  assert.equal(
    /DEMO_ROUTES/.test(mapPanelCode),
    false,
    'MapPanel must NOT reference DEMO_ROUTES'
  )
  assert.equal(
    /GROUP_DEMO_USERS/.test(mapPanelCode),
    false,
    'MapPanel must NOT reference GROUP_DEMO_USERS'
  )
  assert.equal(
    /from '\.\/groupDemo\/groupDemoLogic'/.test(mapPanelCode),
    false,
    'MapPanel must NOT import groupDemoLogic'
  )
  assert.equal(
    /sim_user_1/.test(mapPanelCode),
    false,
    'MapPanel must NOT reference hardcoded sim_user_1'
  )
})

test('2. MapPanel removes fake demo banner, simulation controls, and misleading labels', () => {
  assert.equal(
    /SIMULATED DEMO — NOT LIVE GPS/.test(mapPanelCode),
    false,
    'MapPanel must NOT contain simulated demo banner'
  )
  assert.equal(
    /3 Simulated Users/.test(mapPanelCode),
    false,
    'MapPanel must NOT contain misleading "3 Simulated Users" label'
  )
  assert.equal(
    /group-demo-dashboard-strip/.test(mapPanelCode),
    false,
    'MapPanel must NOT contain simulated demo dashboard strip'
  )
  assert.equal(
    /isDemoPlaying/.test(mapPanelCode),
    false,
    'MapPanel must NOT have simulated movement playback timer or controls'
  )
})

test('3. MapPanel eliminates hardcoded San Francisco coordinates fallback', () => {
  assert.equal(
    /37\.7749/.test(mapPanelCode),
    false,
    'MapPanel must NOT contain hardcoded 37.7749 San Francisco latitude'
  )
  assert.equal(
    /-122\.4194/.test(mapPanelCode),
    false,
    'MapPanel must NOT contain hardcoded -122.4194 San Francisco longitude'
  )
})

/* ==========================================================================
   Part 2: Real Multi-Patient Tracking View
   ========================================================================== */

test('4. MapPanel provides real Multi-Patient Tracking toggle and mode', () => {
  assert.match(
    mapPanelCode,
    /isMultiPatientView,\s*setIsMultiPatientView/,
    'MapPanel must define isMultiPatientView toggle state'
  )
  assert.match(
    mapPanelCode,
    /Single Patient/,
    'Toolbar must have Single Patient toggle option'
  )
  assert.match(
    mapPanelCode,
    /Multi-Patient View/,
    'Toolbar must have Multi-Patient View toggle option'
  )
})

test('5. MapPanel supports dynamic addition and removal of actual patient IDs', () => {
  assert.match(
    mapPanelCode,
    /multiPatientIds,\s*setMultiPatientIds/,
    'Must maintain list of tracked patient IDs'
  )
  assert.match(
    mapPanelCode,
    /handleAddMultiPatient/,
    'Must provide handler to add any patient ID'
  )
  assert.match(
    mapPanelCode,
    /handleRemoveMultiPatient/,
    'Must provide handler to remove patient ID'
  )
  assert.match(
    mapPanelCode,
    /id="multi-patient-input"/,
    'Must provide accessible input to enter patient ID'
  )
})

test('6. MapPanel independently fetches real recorded history per patient from API', () => {
  assert.match(
    mapPanelCode,
    /fetch\(`\/api\/risk-history\/\$\{encodeURIComponent\(pid\)\}\?limit=100`\)/,
    'Must query backend API for each patient ID independently'
  )
  assert.match(
    mapPanelCode,
    /Historical recorded trajectory · Not live GPS/i,
    'Must clearly indicate data source is historical recorded trajectories, not live GPS'
  )
})

test('7. MapPanel applies distinct color palette and marker styles per patient ID', () => {
  assert.match(
    mapPanelCode,
    /MULTI_PATIENT_PALETTE/,
    'Must define multi-patient color palette'
  )
  assert.match(
    mapPanelCode,
    /createMultiPatientMarkerIcon/,
    'Must create custom markers tied to patient ID'
  )
  assert.match(
    mapPanelCode,
    /positions=\{p\.coords\}[\s\S]*?color:\s*p\.color/,
    'Must render real trajectory Polyline using patient distinct color'
  )
})

test('8. Multi-patient bounds fitting automatically spans all loaded patient coordinates', () => {
  assert.match(
    mapPanelCode,
    /function MapMultiPatientBoundsController\(\{[\s\S]*?bounds[\s\S]*?\}\)/,
    'Must define MapMultiPatientBoundsController'
  )
  assert.match(
    mapPanelCode,
    /map\.fitBounds\(bounds,\s*\{ padding: \[\d+,\s*\d+\],\s*maxZoom: \d+ \}\)/,
    'Controller must fit map to patient bounds'
  )
  assert.match(
    mapPanelCode,
    /Fit All Patients/,
    'Toolbar must offer "Fit All Patients" button in multi-patient mode'
  )
})

test('9. MapPanel shows clean empty state when no patient coordinates exist (never SF fallback)', () => {
  assert.match(
    mapPanelCode,
    /No recorded location data for this patient/,
    'Must show exact empty state message for single patient with no records'
  )
  assert.match(
    mapPanelCode,
    /No recorded location data for selected patients/,
    'Must show clear empty state for multi-patient mode when no points exist'
  )
})

/* ==========================================================================
   Part 3: Spatial Math & Geofence Logic Preserved
   ========================================================================== */

test('10. Haversine distance computes accurate meter distances', () => {
  const p1 = [39.9847, 116.3184]
  const p2 = [39.9847, 116.3184]
  assert.equal(haversineMeters(p1, p2), 0, 'Same point distance must be 0')

  // Distance of approx 100 meters northward (~0.0009 degrees latitude)
  const pNorth = [39.9856, 116.3184]
  const dist = haversineMeters(p1, pNorth)
  assert.ok(dist >= 95 && dist <= 105, `Distance should be ~100m, got ${dist}`)
})

test('11. Safe Zone boundary check correctly classifies coordinates', () => {
  const center = DEFAULT_GEOFENCE.center
  assert.ok(isInsideGeofence(center, DEFAULT_GEOFENCE), 'Geofence center must be inside')

  // Far coordinate (~20km away)
  const outsideCoord = [center[0] + 0.2, center[1] + 0.2]
  assert.equal(isInsideGeofence(outsideCoord, DEFAULT_GEOFENCE), false, 'Far point must be outside')
})

test('12. CSS styling exists for multi-patient tracking UI components', () => {
  assert.match(cssCode, /\.multi-patient-dashboard-strip/, 'Must have .multi-patient-dashboard-strip CSS')
  assert.match(cssCode, /\.multi-patient-chip/, 'Must have .multi-patient-chip CSS')
  assert.match(cssCode, /\.multi-patient-input/, 'Must have .multi-patient-input CSS')
  assert.match(cssCode, /\.multi-patient-add-btn/, 'Must have .multi-patient-add-btn CSS')
})
