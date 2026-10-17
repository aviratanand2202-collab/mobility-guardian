/**
 * Unit & Contract Verification for Patient Consent Setup UI (§2.7 Compliance)
 *
 * Requirements Verified:
 * 1. Schema Compliance: POST /api/consent/record schema conforms to spec §2.7.
 * 2. Explicit Non-Negotiables: No pre-checked boxes, no fabricated consent, no assumed incapacity.
 * 3. Capacity Assessment: Requires explicit selection of patient cognitive/legal capacity.
 * 4. Conditional Patient Consent: Explicit patient consent checkbox shown and enforced when capacity exists.
 * 5. Data Retention Acknowledgment: Explicit policy acknowledgment required.
 * 6. Caregiver Dashboard Integration:
 *    - Queries GET /api/consent/{user_id}.
 *    - Displays '⚠️ Patient Consent Required' pill when missing.
 *    - Displays '✓ Patient Consent on File' pill when granted.
 *    - Opens PatientConsentModal on request.
 * 7. Trusted Companion Integration:
 *    - Displays consent warning notice and setup action before inviting companion.
 *    - Offers inline resolution button if backend 403 consent error occurs.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const modalPath = path.resolve(__dirname, '../src/components/caregiver/PatientConsentModal.jsx')
const modalCode = fs.readFileSync(modalPath, 'utf8')

const caregiverDashboardPath = path.resolve(__dirname, '../src/components/caregiver/CaregiverDashboard.jsx')
const caregiverDashboardCode = fs.readFileSync(caregiverDashboardPath, 'utf8')

const companionPanelPath = path.resolve(__dirname, '../src/components/companion/TrustedCompanionPanel.jsx')
const companionPanelCode = fs.readFileSync(companionPanelPath, 'utf8')

/* ==========================================================================
   Part 1: Patient Consent Modal Compliance & Validation Rules
   ========================================================================== */

test('1. PatientConsentModal initializes all consent checkboxes to unchecked (false) and capacity to null', () => {
  // Check default states in component
  assert.match(
    modalCode,
    /const \[guardianConsent, setGuardianConsent\] = useState\(false\)/,
    'Guardian consent checkbox must default to false'
  )
  assert.match(
    modalCode,
    /const \[endUserApplicable, setEndUserApplicable\] = useState\(null\)/,
    'Patient capacity must default to null (must not assume capacity or lack of capacity)'
  )
  assert.match(
    modalCode,
    /const \[endUserConsent, setEndUserConsent\] = useState\(false\)/,
    'Patient consent checkbox must default to false'
  )
  assert.match(
    modalCode,
    /const \[dataRetentionAck, setDataRetentionAck\] = useState\(false\)/,
    'Data retention acknowledgment must default to false'
  )
})

test('2. PatientConsentModal strictly validates all required §2.7 fields before submission', () => {
  // Guardian ID validation
  assert.match(
    modalCode,
    /if \(!guardianId\.trim\(\)\)/,
    'Must validate non-empty Guardian / Caregiver ID'
  )
  // Explicit Guardian Consent validation
  assert.match(
    modalCode,
    /if \(!guardianConsent\)/,
    'Must validate explicit Guardian consent'
  )
  // Explicit Capacity Assessment validation
  assert.match(
    modalCode,
    /if \(endUserApplicable === null\)/,
    'Must enforce explicit selection of patient capacity assessment'
  )
  // End-user consent required when applicable
  assert.match(
    modalCode,
    /if \(endUserApplicable && !endUserConsent\)/,
    'Must require explicit patient consent when patient retains capacity'
  )
  // Data retention acknowledgment validation
  assert.match(
    modalCode,
    /if \(!dataRetentionAck\)/,
    'Must require data retention acknowledgment'
  )
})

test('3. PatientConsentModal builds canonical schema payload and calls POST /api/consent/record', () => {
  assert.match(
    modalCode,
    /\/api\/consent\/record/,
    'Must POST to /api/consent/record endpoint'
  )
  assert.match(
    modalCode,
    /guardian_consent:\s*\{[\s\S]*?granted:\s*true[\s\S]*?guardian_id:/,
    'Payload must conform to guardian_consent schema'
  )
  assert.match(
    modalCode,
    /end_user_consent_applicable:\s*endUserApplicable/,
    'Payload must specify end_user_consent_applicable'
  )
  assert.match(
    modalCode,
    /data_retention_ack:\s*true/,
    'Payload must specify data_retention_ack: true'
  )
})

/* ==========================================================================
   Part 2: Caregiver Dashboard Consent Status & Modal Wiring
   ========================================================================== */

test('4. CaregiverDashboard queries GET /api/consent/{userId} on mount and user change', () => {
  assert.match(
    caregiverDashboardCode,
    /import PatientConsentModal from '\.\/PatientConsentModal'/,
    'CaregiverDashboard must import PatientConsentModal'
  )
  assert.match(
    caregiverDashboardCode,
    /\/api\/consent\/\$\{encodeURIComponent\(targetId\.trim\(\)\)\}/,
    'CaregiverDashboard must fetch consent record by userId'
  )
  assert.match(
    caregiverDashboardCode,
    /fetchPatientConsent\(userId\)/,
    'CaregiverDashboard must fetch patient consent on userId change'
  )
})

test('5. CaregiverDashboard displays consent status pill and opens modal when clicked', () => {
  assert.match(
    caregiverDashboardCode,
    /Patient Consent Required/,
    'Dashboard header must display "Patient Consent Required" when missing'
  )
  assert.match(
    caregiverDashboardCode,
    /Patient Consent on File/,
    'Dashboard header must display "Patient Consent on File" when granted'
  )
  assert.match(
    caregiverDashboardCode,
    /onClick=\{\(\) => setShowConsentModal\(true\)\}/,
    'Clicking unconsented status pill must open consent setup modal'
  )
  assert.match(
    caregiverDashboardCode,
    /<PatientConsentModal[\s\S]*?userId=\{userId\}[\s\S]*?onClose=\{[\s\S]*?\}[\s\S]*?onConsentRecorded=\{handleConsentRecorded\}/,
    'Dashboard must render PatientConsentModal with close and recorded handlers'
  )
})

/* ==========================================================================
   Part 3: Trusted Companion Panel Integration
   ========================================================================== */

test('6. TrustedCompanionPanel accepts patientConsent and offers setup action', () => {
  assert.match(
    companionPanelCode,
    /patientConsent/,
    'TrustedCompanionPanel must accept patientConsent prop'
  )
  assert.match(
    companionPanelCode,
    /onOpenConsentModal/,
    'TrustedCompanionPanel must accept onOpenConsentModal callback'
  )
  assert.match(
    companionPanelCode,
    /Patient Consent Required \(§2\.7\)/,
    'TrustedCompanionPanel must display clear consent notice when unconsented'
  )
  assert.match(
    companionPanelCode,
    /Set Up Patient Consent Now/i,
    'TrustedCompanionPanel must provide inline resolution button if 403 consent error occurs'
  )
})
