/**
 * Regression Test Suite for Phase 3B: Safe-Zone-Only Map Rendering
 *
 * Verifies:
 * 1. Safe-zone-only map mounting when GPS/history is absent but valid safe zones exist.
 * 2. Strict avoidance of patient location marker at fallback safe-zone center.
 * 3. Exact state labeling: "Awaiting GPS — safe zones shown; patient location unknown."
 * 4. Normal GPS and historical location precedence preservation.
 * 5. Preservation of empty-state when neither GPS nor safe zones exist.
 * 6. Safe handling of disabled zones (is_enabled === false) and invalid/missing coordinates.
 * 7. Multi-patient tracking independence.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

const mapPanelCode = fs.readFileSync(
  path.resolve('frontend/src/components/MapPanel.jsx'),
  'utf8'
)

/* ==========================================================================
   Part 1: Precedence & Fallback Center Calculation
   ========================================================================== */

test('1. Normal GPS and historical position precedence is preserved', () => {
  // activePosition must use currentLocation first, then latest locationHistory, then null
  assert.match(
    mapPanelCode,
    /const activePosition =\s*currentLocation && currentLocation\.lat != null && currentLocation\.lng != null\s*\?\s*\[currentLocation\.lat,\s*currentLocation\.lng\]\s*:\s*locationHistory\.length > 0\s*\?\s*locationHistory\[locationHistory\.length - 1\]\s*:\s*null/,
    'activePosition must prioritize current GPS over historical location and remain null when both are absent'
  )

  // Map center determination preserves activePosition > locationHistory[0] > safeZoneCenter
  assert.match(
    mapPanelCode,
    /const center = isMultiPatientView\s*\?\s*\(allMultiPatientCoords\.length > 0 \? allMultiPatientCoords\[0\] : null\)\s*:\s*\(activePosition \|\| \(locationHistory\.length > 0 \? locationHistory\[0\] : safeZoneCenter\)\)/,
    'Map center must strictly prioritize activePosition, then history, then fallback to safeZoneCenter'
  )
})

test('2. Filters out disabled zones and invalid/missing coordinates safely', () => {
  // Must filter out disabled zones (is_enabled === false)
  assert.match(
    mapPanelCode,
    /zone\.is_enabled === false/,
    'Must filter out disabled safe zones'
  )

  // Must validate coordinates are numbers and not NaN
  assert.match(
    mapPanelCode,
    /typeof zone\.center_lat === 'number' &&\s*!isNaN\(zone\.center_lat\)/,
    'Must check center_lat is a finite number'
  )
  assert.match(
    mapPanelCode,
    /typeof zone\.center_lng === 'number' &&\s*!isNaN\(zone\.center_lng\)/,
    'Must check center_lng is a finite number'
  )

  // Must validate polygonal vertices have at least 3 points
  assert.match(
    mapPanelCode,
    /zone\.geometry_type === 'polygon'[\s\S]*?zone\.polygon_vertices\.length >= 3/,
    'Must validate polygonal safe zones have at least 3 vertices'
  )
})

/* ==========================================================================
   Part 2: Safe-Zone-Only Mounting & Empty-State Preservation
   ========================================================================== */

test('3. Mounts map when safe zone exists without triggering empty state', () => {
  // Empty state in Caregiver mode must require BOTH no activePosition AND no safeZoneCenter
  assert.match(
    mapPanelCode,
    /!isMultiPatientView && isCaregiverMode && !activePosition && !safeZoneCenter \? \(/,
    'Caregiver empty state must check !safeZoneCenter before suppressing map'
  )

  // Empty state in Technical mode must require BOTH no activePosition AND no safeZoneCenter
  assert.match(
    mapPanelCode,
    /!isMultiPatientView && !activePosition && !safeZoneCenter \? \(/,
    'Technical empty state must check !safeZoneCenter before suppressing map'
  )

  // When neither usable location nor safe zone exists, clean empty state is preserved
  assert.match(
    mapPanelCode,
    /No recorded location data for this patient/,
    'Must preserve clean empty state description when neither location nor safe zone exists'
  )
})

test('4. Does NOT render a patient location marker at fallback safe-zone center', () => {
  // Patient marker MUST be conditioned strictly on activePosition, never on safeZoneCenter or center
  assert.match(
    mapPanelCode,
    /\{activePosition && \(\s*<Marker position=\{activePosition\}/,
    'Patient marker must only be rendered when activePosition is truthy'
  )
  assert.doesNotMatch(
    mapPanelCode,
    /<Marker position=\{safeZoneCenter\}/,
    'Must never render patient marker at safeZoneCenter'
  )
  assert.doesNotMatch(
    mapPanelCode,
    /<Marker position=\{center\}/,
    'Must never render patient marker at generic center'
  )
})

/* ==========================================================================
   Part 3: State Labeling & Accessibility
   ========================================================================== */

test('5. Clearly labels state "Awaiting GPS — safe zones shown; patient location unknown."', () => {
  const expectedLabel = 'Awaiting GPS — safe zones shown; patient location unknown.'

  // Corner badge renders exact label in safe-zone-only mode
  assert.match(
    mapPanelCode,
    /map-corner-badge-safezone-only[\s\S]*?Awaiting GPS — safe zones shown; patient location unknown\./,
    'Corner badge must display exact state label'
  )

  // Sector badge in header renders label
  assert.match(
    mapPanelCode,
    /isSafeZoneOnly[\s\S]*?Awaiting GPS — safe zones shown; patient location unknown\./,
    'Header sector badge must display exact state label'
  )

  // Map overlay notice renders label
  assert.match(
    mapPanelCode,
    /safeZoneCenter\s*\?\s*['"]Awaiting GPS — safe zones shown; patient location unknown\.['"]/,
    'Map overlay notice must display exact state label when safeZoneCenter is active'
  )
})

/* ==========================================================================
   Part 4: Recenter & View Management in Safe-Zone-Only Mode
   ========================================================================== */

test('6. Recenter button targets safe-zone center when in safe-zone-only mode', () => {
  // handleRecenter flies to safeZoneCenter when activePosition is absent
  assert.match(
    mapPanelCode,
    /\} else if \(safeZoneCenter\) \{\s*setSelectedPointIndex\(null\)\s*setViewTarget\(\{ coords: safeZoneCenter, mode: 'fly', key: Date\.now\(\) \}\)/,
    'handleRecenter must target safeZoneCenter when activePosition is absent'
  )

  // Recenter button remains enabled if safeZoneCenter is present
  assert.match(
    mapPanelCode,
    /disabled=\{!activePosition && !safeZoneCenter\}/,
    'Recenter button must be enabled when safeZoneCenter is available'
  )
})

/* ==========================================================================
   Part 5: Multi-Patient View Independence
   ========================================================================== */

test('7. Multi-patient tracking behavior remains independent and unchanged', () => {
  assert.match(
    mapPanelCode,
    /const isSafeZoneOnly = !isMultiPatientView && !hasUsableLocation && Boolean\(safeZoneCenter\)/,
    'isSafeZoneOnly must be strictly disabled when isMultiPatientView is active'
  )
  assert.match(
    mapPanelCode,
    /isMultiPatientView && allMultiPatientCoords\.length === 0/,
    'Multi-patient empty state condition must remain unchanged'
  )
})
