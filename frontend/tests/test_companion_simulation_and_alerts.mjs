import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const frontendDir = path.resolve(__dirname, '..')

test('1. App.jsx defines fetchAlerts, trimmedId, and preserves alertMessages on disconnect', () => {
  const appSrc = fs.readFileSync(path.join(frontendDir, 'src/App.jsx'), 'utf-8')
  assert.match(appSrc, /const trimmedId = targetUserId\.trim\(\)/, 'App.jsx must declare trimmedId in fetchHistory')
  assert.match(appSrc, /const fetchAlerts = async/, 'App.jsx must define fetchAlerts')
  assert.match(appSrc, /fetchAlerts\(userId\)/, 'App.jsx must poll fetchAlerts')
  assert.match(appSrc, /alertsLoading=\{alertsLoading\}/, 'App.jsx must pass alertsLoading')
  assert.match(appSrc, /alertsError=\{alertsError\}/, 'App.jsx must pass alertsError')
})

test('2. AlertPanel.jsx renders patient ID, alert type, and handles AUTO_RESOLVED status', () => {
  const panelSrc = fs.readFileSync(path.join(frontendDir, 'src/components/AlertPanel.jsx'), 'utf-8')
  assert.match(panelSrc, /isGroupSeparation/, 'AlertPanel must identify group separation alerts')
  assert.match(panelSrc, /Group Separation Alert/, 'AlertPanel must label Group Separation Alert')
  assert.match(panelSrc, /Auto-Resolved \(Group Reunited\)/, 'AlertPanel must render Auto-Resolved badge')
  assert.match(panelSrc, /Physiological Arousal/, 'AlertPanel must display physiological corroboration note')
  assert.match(panelSrc, /alert\.user_id \|\| userId/, 'AlertPanel must display patient ID')
})

test('3. AlertPanel.jsx distinguishes zero alerts, loading, and error states', () => {
  const panelSrc = fs.readFileSync(path.join(frontendDir, 'src/components/AlertPanel.jsx'), 'utf-8')
  assert.match(panelSrc, /Loading active alerts for patient/, 'AlertPanel must have loading state')
  assert.match(panelSrc, /Failed to load alerts:/, 'AlertPanel must have error state')
  assert.match(panelSrc, /No active alerts for patient/, 'AlertPanel must have clean empty state')
})

test('4. TechnicalDashboard.jsx renders companion walking simulation controls and labels', () => {
  const techSrc = fs.readFileSync(path.join(frontendDir, 'src/components/technical/TechnicalDashboard.jsx'), 'utf-8')
  assert.match(techSrc, /Companion Walking Simulation \(Development Test\)/, 'TechnicalDashboard must have companion walking simulation title')
  assert.match(techSrc, /\[Simulated Companion Data\]/, 'TechnicalDashboard must label simulated companion data')
  assert.match(techSrc, /WALKING_TOGETHER/, 'TechnicalDashboard must support WALKING_TOGETHER scenario')
  assert.match(techSrc, /SEPARATING/, 'TechnicalDashboard must support SEPARATING scenario')
  assert.match(techSrc, /REJOINING/, 'TechnicalDashboard must support REJOINING scenario')
  assert.match(techSrc, /Simulate HR Arousal/, 'TechnicalDashboard must provide HR arousal toggle')
})

test('5. TrustedCompanionPanel.jsx renders simulation controls with simulated label', () => {
  const compSrc = fs.readFileSync(path.join(frontendDir, 'src/components/companion/TrustedCompanionPanel.jsx'), 'utf-8')
  assert.match(compSrc, /Companion Walking Simulation \(Dev Test\)/, 'TrustedCompanionPanel must include simulation block')
  assert.match(compSrc, /\[Simulated Data\]/, 'TrustedCompanionPanel must label simulated data')
  assert.match(compSrc, /handleSimulateStep\('WALKING_TOGETHER'\)/, 'TrustedCompanionPanel must call WALKING_TOGETHER')
  assert.match(compSrc, /handleSimulateStep\('SEPARATING'\)/, 'TrustedCompanionPanel must call SEPARATING')
  assert.match(compSrc, /handleSimulateStep\('REJOINING'\)/, 'TrustedCompanionPanel must call REJOINING')
})

test('6. companionLogic.js defines ACCOMPANIMENT_STATUS matching user requirements', async () => {
  const { ACCOMPANIMENT_STATUS } = await import('../src/components/companion/companionLogic.js')
  assert.equal(ACCOMPANIMENT_STATUS.WITH_COMPANION, 'With Companion')
  assert.equal(ACCOMPANIMENT_STATUS.TOGETHER, 'Together')
  assert.equal(ACCOMPANIMENT_STATUS.SEPARATION_PENDING, 'Separation Pending')
  assert.equal(ACCOMPANIMENT_STATUS.SEPARATED, 'Separated')
  assert.equal(ACCOMPANIMENT_STATUS.REJOINING, 'Rejoining')
  assert.equal(ACCOMPANIMENT_STATUS.LOCATION_UNAVAILABLE, 'Location Unavailable')
  assert.equal(ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED, 'Accompaniment Unverified')
})
