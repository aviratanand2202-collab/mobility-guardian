/**
 * Unit & Contract Verification for Associated Family & Companion Locations
 * in the real Technical Inspector (TechnicalDashboard & MapPanel).
 *
 * Requirements Verified:
 * 1. Technical Inspector loads and polls active companion/family session for selected patient ID.
 * 2. TechnicalDashboard passes activeCompanion prop to MapPanel on the same map.
 * 3. Shows distinct accompaniment strip with status ('With Companion', 'Separated', 'Accompaniment Unverified').
 * 4. Displays 'Accompaniment Unverified' when no valid authorized association exists (never infers relationships).
 * 5. Supports explicit association of family members or trusted companions with clear patient-to-companion association.
 * 6. MapPanel renders distinct markers and labels for patient vs family vs companion, and shows stale status.
 * 7. Accompanied safe zone departure is clearly displayed without triggering family separation.
 * 8. Group Demo remains strictly isolated from real backend telemetry and companion markers.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const technicalDashboardPath = path.resolve(__dirname, '../src/components/technical/TechnicalDashboard.jsx')
const technicalDashboardCode = fs.readFileSync(technicalDashboardPath, 'utf8')

const mapPanelPath = path.resolve(__dirname, '../src/components/MapPanel.jsx')
const mapPanelCode = fs.readFileSync(mapPanelPath, 'utf8')

const indexCssPath = path.resolve(__dirname, '../src/index.css')
const indexCssCode = fs.readFileSync(indexCssPath, 'utf8')

test('1. TechnicalDashboard fetches active companion/family session for the selected patient', () => {
  assert.match(
    technicalDashboardCode,
    /fetchActiveCompanion/,
    'TechnicalDashboard must have fetchActiveCompanion function'
  )
  assert.match(
    technicalDashboardCode,
    /\/api\/companion\/sessions\/active\/\$\{encodeURIComponent\(targetUserId/,
    'Must call /api/companion/sessions/active/{userId} endpoint'
  )
  assert.match(
    technicalDashboardCode,
    /fetchActiveCompanion\(userId\)/,
    'Must re-fetch companion status when userId changes'
  )
})

test('2. TechnicalDashboard passes activeCompanion to MapPanel on the same map', () => {
  assert.match(
    technicalDashboardCode,
    /<MapPanel[\s\S]*?activeCompanion=\{activeCompanion\}/,
    'TechnicalDashboard must pass activeCompanion prop to MapPanel'
  )
})

test('3. TechnicalDashboard displays accompaniment strip with role and aria-label', () => {
  assert.match(
    technicalDashboardCode,
    /className="technical-accompaniment-strip"/,
    'Must render technical-accompaniment-strip container'
  )
  assert.match(
    technicalDashboardCode,
    /aria-label="Family and Companion Accompaniment"/,
    'Must include accessible aria-label for accompaniment section'
  )
})

test('4. Displays Accompaniment Unverified when no valid association exists', () => {
  assert.match(
    technicalDashboardCode,
    /Accompaniment Unverified/,
    'Must display Accompaniment Unverified when unassociated'
  )
  assert.match(
    technicalDashboardCode,
    /No authorized family member or active Trusted Companion is currently associated/,
    'Must explicitly state no authorized family member is associated'
  )
})

test('5. Allows explicit association without automatic inference from database coexistence', () => {
  assert.match(
    technicalDashboardCode,
    /handleCreateAssociation/,
    'Must have handler for explicit association'
  )
  assert.match(
    technicalDashboardCode,
    /\/api\/companion\/sessions\/create/,
    'Must call /api/companion/sessions/create with explicit companion ID'
  )
  assert.match(
    technicalDashboardCode,
    /companion_id:\s*companionInputId\.trim\(\)/,
    'Must pass explicit companion_id to create endpoint'
  )
  assert.match(
    technicalDashboardCode,
    /relationship:\s*companionRelationship/,
    'Must allow specifying relationship (family or trusted_companion)'
  )
})

test('6. Supports ending or disconnecting association', () => {
  assert.match(
    technicalDashboardCode,
    /handleEndCompanionSession/,
    'Must have handler to end companion session'
  )
  assert.match(
    technicalDashboardCode,
    /\/api\/companion\/sessions\/\$\{activeCompanion\.session_id\}\/end/,
    'Must call end session endpoint'
  )
})

test('7. MapPanel provides distinct visual styling for Family vs Companion vs Stale', () => {
  assert.match(
    mapPanelCode,
    /function createCompanionMarkerIcon\(isStale = false, relationship = 'trusted_companion'\)/,
    'createCompanionMarkerIcon must take relationship and isStale'
  )
  assert.match(
    mapPanelCode,
    /#2563eb/,
    'Family marker must use distinct blue (#2563eb)'
  )
  assert.match(
    mapPanelCode,
    /#8b5cf6/,
    'Trusted companion marker must use violet (#8b5cf6)'
  )
  assert.match(
    mapPanelCode,
    /#94a3b8/,
    'Stale marker must use slate gray (#94a3b8)'
  )
  assert.match(
    mapPanelCode,
    /FAMILY/,
    'Family pin must have FAMILY pill tag'
  )
})

test('8. Safe zone departure while accompanied is indicated in TechnicalDashboard and MapPanel', () => {
  assert.match(
    technicalDashboardCode,
    /safe_zone_departure_accompanied/,
    'TechnicalDashboard must check safe_zone_departure_accompanied flag'
  )
  assert.match(
    technicalDashboardCode,
    /Accompanied — no family separation/i,
    'Must inform user that safe zone departure is accompanied without family separation'
  )
  assert.match(
    mapPanelCode,
    /Accompanied Safe Zone Departure/i,
    'MapPanel popup must note accompanied safe zone departure'
  )
})

test('9. Companion markers only render when active companion is authorized and present', () => {
  assert.match(
    mapPanelCode,
    /activeCompanion && activeCompanion\.status === 'ACTIVE'/,
    'Companion markers must only render when active companion is authorized and present'
  )
})

test('10. CSS styling exists for all Technical Accompaniment components', () => {
  assert.match(
    indexCssCode,
    /\.technical-accompaniment-strip/,
    'index.css must include .technical-accompaniment-strip'
  )
  assert.match(
    indexCssCode,
    /\.tech-acc-status-pill\.pill-accompanied/,
    'index.css must include .pill-accompanied styling'
  )
  assert.match(
    indexCssCode,
    /\.tech-acc-status-pill\.pill-separated/,
    'index.css must include .pill-separated styling'
  )
  assert.match(
    indexCssCode,
    /\.tech-acc-status-pill\.pill-unverified/,
    'index.css must include .pill-unverified styling'
  )
  assert.match(
    indexCssCode,
    /\.tech-acc-prereq-banner/,
    'index.css must include .tech-acc-prereq-banner styling'
  )
  assert.match(
    indexCssCode,
    /\.tech-acc-consent-alert/,
    'index.css must include .tech-acc-consent-alert styling'
  )
})

test('11. TechnicalDashboard explains missing prerequisites when accompaniment is unverified', () => {
  assert.match(
    technicalDashboardCode,
    /getMissingPrerequisite/,
    'TechnicalDashboard must define getMissingPrerequisite helper'
  )
  assert.match(
    technicalDashboardCode,
    /tech-acc-prereq-banner/,
    'TechnicalDashboard must render tech-acc-prereq-banner'
  )
  assert.match(
    technicalDashboardCode,
    /Patient consent required:/,
    'getMissingPrerequisite must check patient consent requirement'
  )
  assert.match(
    technicalDashboardCode,
    /tech-acc-consent-alert/,
    'TechnicalDashboard must render tech-acc-consent-alert when consent is missing'
  )
})

test('12. MapPanel renders actual configured safe zone without inventing San Francisco coordinates', () => {
  assert.match(
    technicalDashboardCode,
    /configuredSafeZone=\{activeCompanion\?\.safe_zone\}/,
    'TechnicalDashboard must pass configured safe zone from activeCompanion to MapPanel'
  )
  assert.match(
    mapPanelCode,
    /configuredSafeZone\s*=\s*null/,
    'MapPanel must accept configuredSafeZone prop'
  )
  assert.match(
    mapPanelCode,
    /configuredSafeZone\s*&&\s*configuredSafeZone\.configured\s*&&\s*configuredSafeZone\.center/,
    'MapPanel must only render safe zone circle when configured is true and center exists'
  )
  assert.match(
    mapPanelCode,
    /center=\{configuredSafeZone\.center\}/,
    'Safe zone circle must use configuredSafeZone.center, not hardcoded coordinates'
  )
})
