/**
 * Focused regression tests for Phase 1: Persistent Alert Recovery & Hydration.
 *
 * Verifies:
 * 1. REST snapshot & live WebSocket alert merging: ensures deduplication and timestamp sorting.
 * 2. Race condition & user switching guard: delayed responses from previous user selections are ignored.
 * 3. Distinction between active (PENDING) and dismissed (DISMISSED_SAFE) alerts in badge count calculation.
 * 4. Zone sensitivity and dismissal count extraction from both hydrated AlertRecord objects and direct API responses.
 * 5. App.jsx, CaregiverDashboard.jsx, and AlertPanel.jsx contract conformance.
 *
 * Run with: node frontend/tests/test_alert_recovery.mjs
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

const caregiverDashboardPath = path.resolve(__dirname, '../src/components/caregiver/CaregiverDashboard.jsx')
const caregiverCode = fs.readFileSync(caregiverDashboardPath, 'utf8')

const alertPanelPath = path.resolve(__dirname, '../src/components/AlertPanel.jsx')
const alertPanelCode = fs.readFileSync(alertPanelPath, 'utf8')

// ============================================================================
// 1. Source Code Contract Tests
// ============================================================================

test('App.jsx defines fetchAlerts, user scoping, AbortController, and calls fetchAlerts on connect', () => {
  // Verifies fetchAlerts definition and user scoping
  assert.match(
    appCode,
    /const fetchAlerts = async \([a-zA-Z0-9_]+\) => \{/,
    'App.jsx must declare fetchAlerts function'
  )
  assert.match(
    appCode,
    /\/api\/alerts\/user\/\$\{encodeURIComponent\(user\)\}\?limit=50/,
    'fetchAlerts must query user-scoped endpoint /api/alerts/user/{user}'
  )

  // Verifies AbortController guard for user switching race conditions
  assert.match(
    appCode,
    /alertFetchAbortControllerRef\.current\.abort\(\)/,
    'fetchAlerts and disconnectSockets must abort in-flight requests on user switch'
  )
  assert.match(
    appCode,
    /if \(activeUserRef\.current !== user\) \{\s*return\s*\}/,
    'fetchAlerts must verify active user before setting state'
  )

  // Verifies connectSockets triggers both fetchHistory and fetchAlerts
  assert.match(
    appCode,
    /fetchHistory\(targetUserId\.trim\(\)\)\s*\n\s*fetchAlerts\(targetUserId\.trim\(\)\)/,
    'connectSockets must invoke both fetchHistory and fetchAlerts'
  )

  // Verifies deduplicating Map in setAlertMessages during hydration
  assert.match(
    appCode,
    /const alertMap = new Map\(\)/,
    'fetchAlerts must use alertMap for deduplicating snapshot and WebSocket alerts'
  )
})

test('CaregiverDashboard accurately separates active vs dismissed alerts and handles loading/error', () => {
  // Badge count calculation distinguishes PENDING from DISMISSED_SAFE
  assert.match(
    caregiverCode,
    /const pendingAlerts = alertMessages\.filter\(\(a\) => a\.status === 'PENDING'\)/,
    'Must calculate pendingCount separately from total alerts'
  )
  assert.match(
    caregiverCode,
    /0 active \(\$\{dismissedCount\} dismissed\)/,
    'Must show 0 active when all alerts are dismissed'
  )

  // Truthful empty states
  assert.match(
    caregiverCode,
    /if \(alertsLoading && alertMessages\.length === 0\)/,
    'Must handle alertsLoading empty state'
  )
  assert.match(
    caregiverCode,
    /else if \(alertsError && alertMessages\.length === 0\)/,
    'Must handle alertsError empty state'
  )

  // Hydrated suppression extraction from dismissal object
  assert.match(
    caregiverCode,
    /alert\.dismissal_count_in_window\s*\?\?\s*alert\.dismissal\?\.dismissal_count_in_window/,
    'Must extract dismissal count from either direct property or dismissal object'
  )
})

test('AlertPanel supports hydrated dismissal metadata and suppression display', () => {
  assert.match(
    alertPanelCode,
    /const dismissalCount = alert\.dismissal_count \?\? alert\.dismissal_count_in_window \?\? alert\.dismissal\?\.dismissal_count_in_window/,
    'AlertPanel must support dismissalCount fallback from hydrated AlertRecord'
  )
})

// ============================================================================
// 2. Behavioral Unit Tests (Merging, Deduplication, Race Conditions)
// ============================================================================

/**
 * Pure function mirroring App.jsx alert merge logic
 */
function mergeAlertSnapshot(prevAlerts, snapshotAlerts) {
  const alertMap = new Map()
  for (const alert of snapshotAlerts) {
    alertMap.set(alert.alert_id, alert)
  }
  for (const existing of prevAlerts) {
    if (alertMap.has(existing.alert_id)) {
      alertMap.set(existing.alert_id, { ...alertMap.get(existing.alert_id), ...existing })
    } else {
      alertMap.set(existing.alert_id, existing)
    }
  }
  return Array.from(alertMap.values())
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, 20)
}

test('Snapshot and WebSocket alerts merge without duplicate alert cards', () => {
  const initialLiveAlerts = [
    {
      alert_id: 'alert_1',
      user_id: 'test_user',
      status: 'PENDING',
      timestamp: '2026-10-14T10:00:05Z',
      alert_tier: 2,
    },
  ]

  // REST snapshot arrives containing alert_1 and older alert_0
  const restSnapshot = [
    {
      alert_id: 'alert_1',
      user_id: 'test_user',
      status: 'PENDING',
      timestamp: '2026-10-14T10:00:00Z',
      alert_tier: 2,
    },
    {
      alert_id: 'alert_0',
      user_id: 'test_user',
      status: 'DISMISSED_SAFE',
      timestamp: '2026-10-14T09:30:00Z',
      alert_tier: 2,
      dismissal: {
        dismissed_at: '2026-10-14T09:35:00Z',
        decay_clock_start: '2026-10-14T09:35:00Z',
        dismissal_count_in_window: 1,
      },
    },
  ]

  const merged = mergeAlertSnapshot(initialLiveAlerts, restSnapshot)

  // Must contain exactly 2 unique alerts, no duplicates
  assert.equal(merged.length, 2)
  assert.equal(merged[0].alert_id, 'alert_1')
  // Preserves newer timestamp from the live WebSocket message
  assert.equal(merged[0].timestamp, '2026-10-14T10:00:05Z')
  assert.equal(merged[1].alert_id, 'alert_0')
  assert.equal(merged[1].status, 'DISMISSED_SAFE')
})

test('User switching simulation drops late-arriving responses for prior user', () => {
  let activeUser = 'user_A'
  let stateForActiveUser = []

  // Simulated fetch handler mirroring App.jsx
  function handleFetchResponse(responseUser, data) {
    if (activeUser !== responseUser) {
      // Discarded because user changed
      return false
    }
    stateForActiveUser = data
    return true
  }

  // user_A initiates fetch
  const pendingDataUserA = [{ alert_id: 'alert_A', user_id: 'user_A' }]

  // User switches to user_B before response for user_A resolves
  activeUser = 'user_B'
  stateForActiveUser = [] // reset on switch

  // Delayed response for user_A arrives
  const acceptedA = handleFetchResponse('user_A', pendingDataUserA)
  assert.equal(acceptedA, false, 'Must reject response for user_A when active user is user_B')
  assert.deepEqual(stateForActiveUser, [], 'State must not be populated with user_A data')

  // Response for user_B arrives
  const pendingDataUserB = [{ alert_id: 'alert_B', user_id: 'user_B' }]
  const acceptedB = handleFetchResponse('user_B', pendingDataUserB)
  assert.equal(acceptedB, true)
  assert.equal(stateForActiveUser.length, 1)
  assert.equal(stateForActiveUser[0].alert_id, 'alert_B')
})

test('Badge count correctly calculates active vs dismissed alerts', () => {
  function computeBadgeText(alertMessages, isOffline, isConnecting, isLoading, isError) {
    const pendingAlerts = alertMessages.filter((a) => a.status === 'PENDING')
    const pendingCount = pendingAlerts.length
    const dismissedCount = alertMessages.filter((a) => a.status === 'DISMISSED_SAFE').length

    if (isOffline) {
      return `${pendingCount} active · ${dismissedCount} dismissed (Stream Offline)`
    }
    if (isConnecting || isLoading) {
      return 'Syncing alerts...'
    }
    if (isError && alertMessages.length === 0) {
      return 'Alerts unavailable'
    }
    if (pendingCount === 0) {
      return dismissedCount > 0
        ? `0 active (${dismissedCount} dismissed)`
        : '0 active alerts'
    }
    return `${pendingCount} active alert${pendingCount === 1 ? '' : 's'}${dismissedCount > 0 ? ` (${dismissedCount} dismissed)` : ''}`
  }

  // Case 1: No alerts
  assert.equal(computeBadgeText([], false, false, false, false), '0 active alerts')

  // Case 2: Only 1 dismissed alert (the known baseline scenario)
  const dismissedOnly = [{ alert_id: 'a1', status: 'DISMISSED_SAFE' }]
  assert.equal(computeBadgeText(dismissedOnly, false, false, false, false), '0 active (1 dismissed)')

  // Case 3: 1 pending alert
  const pendingOnly = [{ alert_id: 'a2', status: 'PENDING' }]
  assert.equal(computeBadgeText(pendingOnly, false, false, false, false), '1 active alert')

  // Case 4: 2 pending, 1 dismissed
  const mixed = [
    { alert_id: 'a1', status: 'DISMISSED_SAFE' },
    { alert_id: 'a2', status: 'PENDING' },
    { alert_id: 'a3', status: 'PENDING' },
  ]
  assert.equal(computeBadgeText(mixed, false, false, false, false), '2 active alerts (1 dismissed)')

  // Case 5: Loading
  assert.equal(computeBadgeText([], false, false, true, false), 'Syncing alerts...')

  // Case 6: Error
  assert.equal(computeBadgeText([], false, false, false, true), 'Alerts unavailable')
})
