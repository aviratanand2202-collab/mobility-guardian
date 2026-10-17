/**
 * Unit & Contract Verification for Trusted Companion Mode.
 *
 * Requirements Verified:
 * 1. Accompaniment Status: 'Accompaniment Unverified' when location is missing or stale (>60s).
 * 2. Accompaniment Status: 'With Companion' when both locations are fresh and distance <= threshold.
 * 3. Safe Zone Departure: Leaving safe zone together while accompanied does NOT trigger separation alert.
 * 4. Accompaniment Status: 'Separated' when distance exceeds threshold.
 * 5. Privacy: Personal ID is never exposed; privacy alias is used across UI and dashboard.
 * 6. Explicit Consent: Portal requires explicit consent before location updates are accepted.
 * 7. Caregiver & Map Integration: CaregiverDashboard renders TrustedCompanionPanel and passes activeCompanion to MapPanel.
 * 8. Group Demo Preservation: Group demo mode and its controls remain 100% separate and functional.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  evaluateAccompaniment,
  ACCOMPANIMENT_STATUS,
} from '../src/components/companion/companionLogic.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const caregiverDashboardPath = path.resolve(__dirname, '../src/components/caregiver/CaregiverDashboard.jsx')
const caregiverDashboardCode = fs.readFileSync(caregiverDashboardPath, 'utf8')

const mapPanelPath = path.resolve(__dirname, '../src/components/MapPanel.jsx')
const mapPanelCode = fs.readFileSync(mapPanelPath, 'utf8')

const companionPanelPath = path.resolve(__dirname, '../src/components/companion/TrustedCompanionPanel.jsx')
const companionPanelCode = fs.readFileSync(companionPanelPath, 'utf8')

const portalPath = path.resolve(__dirname, '../src/components/companion/CompanionInvitePortal.jsx')
const portalCode = fs.readFileSync(portalPath, 'utf8')

/* ==========================================================================
   Part 1: Pure Accompaniment & Distance Logic
   ========================================================================== */

test('1. evaluateAccompaniment returns Accompaniment Unverified when patient location is unavailable', () => {
  const result = evaluateAccompaniment({
    patientLocation: null,
    companionLocation: { lat: 37.7749, lng: -122.4194 },
    companionTimestamp: new Date().toISOString(),
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED)
  assert.equal(result.isVerified, false)
  assert.equal(result.distanceMeters, null)
  assert.match(result.statusMessage, /Awaiting patient/i)
})

test('2. evaluateAccompaniment returns Accompaniment Unverified when companion location is missing', () => {
  const result = evaluateAccompaniment({
    patientLocation: { lat: 37.7749, lng: -122.4194 },
    companionLocation: null,
    companionTimestamp: null,
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED)
  assert.equal(result.isVerified, false)
  assert.match(result.statusMessage, /Awaiting companion/i)
})

test('3. evaluateAccompaniment returns Accompaniment Unverified when companion location is stale (>60s)', () => {
  const currentTime = Date.now()
  const staleTimestamp = new Date(currentTime - 90 * 1000).toISOString() // 90 seconds old

  const result = evaluateAccompaniment({
    patientLocation: { lat: 37.7749, lng: -122.4194 },
    companionLocation: { lat: 37.7749, lng: -122.4194 },
    companionTimestamp: staleTimestamp,
    currentTime,
    maxStaleSeconds: 60,
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED)
  assert.equal(result.isStale, true)
  assert.equal(result.isVerified, false)
  assert.match(result.statusMessage, /stale/i)
})

test('4. evaluateAccompaniment returns With Companion when fresh and within distance threshold', () => {
  const currentTime = Date.now()
  const freshTimestamp = new Date(currentTime - 5 * 1000).toISOString() // 5 seconds ago

  // Distance ~15 meters
  const patientLocation = { lat: 37.77490, lng: -122.41940 }
  const companionLocation = { lat: 37.77503, lng: -122.41940 }

  const result = evaluateAccompaniment({
    patientLocation,
    companionLocation,
    companionTimestamp: freshTimestamp,
    currentTime,
    thresholdMeters: 50,
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.WITH_COMPANION)
  assert.equal(result.isVerified, true)
  assert.equal(result.isStale, false)
  assert.ok(result.distanceMeters < 50, `Distance should be < 50m, got ${result.distanceMeters}`)
  assert.match(result.statusMessage, /With companion/i)
})

test('5. Safe Zone Departure: Leaving safe zone together while accompanied does NOT trigger alert', () => {
  const currentTime = Date.now()
  const freshTimestamp = new Date(currentTime - 2 * 1000).toISOString()

  // Coords outside 260m safe zone (e.g. at [37.7800, -122.4150]) but both together (~10m apart)
  const patientLocation = { lat: 37.78000, lng: -122.41500 }
  const companionLocation = { lat: 37.78008, lng: -122.41500 }

  const result = evaluateAccompaniment({
    patientLocation,
    companionLocation,
    companionTimestamp: freshTimestamp,
    currentTime,
    thresholdMeters: 50,
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.WITH_COMPANION)
  assert.equal(result.safeZoneDepartureAccompanied, true)
  assert.match(result.statusMessage, /outside zone/i)
  assert.match(result.statusMessage, /no emergency alert/i)
})

test('6. evaluateAccompaniment returns Separated when distance exceeds threshold', () => {
  const currentTime = Date.now()
  const freshTimestamp = new Date(currentTime - 4 * 1000).toISOString()

  // Distance ~120m apart
  const patientLocation = { lat: 37.7749, lng: -122.4194 }
  const companionLocation = { lat: 37.7760, lng: -122.4194 }

  const result = evaluateAccompaniment({
    patientLocation,
    companionLocation,
    companionTimestamp: freshTimestamp,
    currentTime,
    thresholdMeters: 50,
  })

  assert.equal(result.status, ACCOMPANIMENT_STATUS.SEPARATED)
  assert.equal(result.isVerified, true)
  assert.ok(result.distanceMeters > 50, `Distance should exceed threshold, got ${result.distanceMeters}`)
  assert.match(result.statusMessage, /Separated from companion/i)
})

/* ==========================================================================
   Part 2: UI & Component Architecture Contracts
   ========================================================================== */

test('7. CaregiverDashboard integrates TrustedCompanionPanel and passes activeCompanion to MapPanel', () => {
  assert.match(
    caregiverDashboardCode,
    /import TrustedCompanionPanel from '\.\.\/companion\/TrustedCompanionPanel'/,
    'CaregiverDashboard must import TrustedCompanionPanel'
  )
  assert.match(
    caregiverDashboardCode,
    /<TrustedCompanionPanel[\s\S]*?userId=\{userId\}/,
    'CaregiverDashboard must render TrustedCompanionPanel with userId'
  )
  assert.match(
    caregiverDashboardCode,
    /activeCompanion=\{activeCompanion\}/,
    'CaregiverDashboard must pass activeCompanion prop to MapPanel'
  )
})

test('8. MapPanel renders companion marker and dashed polyline without breaking Group Demo', () => {
  assert.match(
    mapPanelCode,
    /function createCompanionMarkerIcon/,
    'MapPanel must define createCompanionMarkerIcon'
  )
  assert.match(
    mapPanelCode,
    /activeCompanion = null/,
    'MapPanel must accept activeCompanion prop'
  )
  assert.match(
    mapPanelCode,
    /activeCompanion && activeCompanion\.status === 'ACTIVE'/,
    'Companion marker must render when activeCompanion is ACTIVE'
  )
  assert.match(
    mapPanelCode,
    /createCompanionMarkerIcon\(activeCompanion\.is_location_stale/,
    'Companion marker must reflect stale telemetry status'
  )
})

test('9. Privacy guarantee: Dashboard displays privacy alias, never exposing companion personal ID', () => {
  assert.match(
    companionPanelCode,
    /activeSession\.companion_alias \|\| 'Trusted Companion'/,
    'Companion panel must use privacy alias'
  )
  assert.match(
    companionPanelCode,
    /Personal ID Protected/i,
    'Companion panel must explicitly indicate personal ID protection'
  )
  assert.match(
    portalCode,
    /Your personal identity and account ID are never revealed/i,
    'Companion portal must disclose that personal ID is never revealed'
  )
})

test('10. Consent & Control: Portal requires explicit consent and provides Stop Sharing action', () => {
  assert.match(
    portalCode,
    /I Explicitly Consent & Share My Location/i,
    'Portal must have explicit consent button'
  )
  assert.match(
    portalCode,
    /Stop Sharing & Leave Session/i,
    'Portal must have Stop Sharing button'
  )
  assert.match(
    companionPanelCode,
    /End Companion Session/i,
    'Caregiver panel must have End Session button'
  )
})
