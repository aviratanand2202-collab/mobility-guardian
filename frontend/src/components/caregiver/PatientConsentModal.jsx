import React, { useState } from 'react'

export default function PatientConsentModal({ userId, onClose, onConsentRecorded }) {
  const [guardianId, setGuardianId] = useState('')
  const [guardianConsent, setGuardianConsent] = useState(false)
  const [endUserApplicable, setEndUserApplicable] = useState(null) // null = not selected, true = yes, false = no
  const [endUserConsent, setEndUserConsent] = useState(false)
  const [dataRetentionAck, setDataRetentionAck] = useState(false)
  const [validationError, setValidationError] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setValidationError(null)

    // 1. Validate Guardian ID
    if (!guardianId.trim()) {
      setValidationError('Guardian / Authorised Caregiver ID is required.')
      return
    }

    // 2. Validate Guardian Consent
    if (!guardianConsent) {
      setValidationError('Explicit guardian/caregiver consent must be granted.')
      return
    }

    // 3. Validate End-User Capacity Selection
    if (endUserApplicable === null) {
      setValidationError('Please specify whether the patient retains legal/cognitive capacity to consent.')
      return
    }

    // 4. Validate End-User Consent if applicable
    if (endUserApplicable && !endUserConsent) {
      setValidationError('Patient consent is required when the patient retains capacity to consent.')
      return
    }

    // 5. Validate Data Retention Acknowledgment
    if (!dataRetentionAck) {
      setValidationError('Data retention policy acknowledgment is required per specification §2.7.')
      return
    }

    setIsSubmitting(true)
    const nowIso = new Date().toISOString()

    const consentPayload = {
      user_id: userId.trim(),
      guardian_consent: {
        granted: true,
        granted_at: nowIso,
        guardian_id: guardianId.trim(),
      },
      end_user_consent_applicable: endUserApplicable,
      end_user_consent: endUserApplicable
        ? {
            granted: true,
            granted_at: nowIso,
          }
        : null,
      data_retention_ack: true,
    }

    try {
      const res = await fetch('/api/consent/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(consentPayload),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Server returned HTTP ${res.status}`)
      }

      const resData = await res.json()
      if (onConsentRecorded) {
        onConsentRecorded(consentPayload)
      }
      onClose()
    } catch (err) {
      console.error('Failed to record patient consent:', err)
      setValidationError(`Failed to record consent: ${err.message}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="companion-portal-overlay" role="dialog" aria-modal="true" aria-labelledby="consent-modal-title">
      <div className="companion-portal-modal patient-consent-modal">
        <div className="companion-portal-header">
          <div className="portal-header-icon">🛡️</div>
          <div className="portal-header-text">
            <h2 id="consent-modal-title" className="portal-title">
              Patient Consent &amp; Authorization
            </h2>
            <span className="portal-subtitle">
              Dual-Consent Compliance Framework (§2.7) · Target: <strong>{userId}</strong>
            </span>
          </div>
          <button
            type="button"
            className="portal-close-btn"
            onClick={onClose}
            aria-label="Close consent dialog"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="companion-portal-body">
          {validationError && (
            <div className="portal-error-banner" role="alert">
              ⚠️ {validationError}
            </div>
          )}

          <div className="consent-callout">
            <span className="callout-kicker">MANDATORY PRIVACY COMPLIANCE</span>
            <p className="callout-disclosure" style={{ marginTop: '4px' }}>
              Per medical compliance and system privacy rules, telemetry ingestion and trusted companion invitations are rejected until an authorized consent record is filed for <strong>{userId}</strong>.
            </p>
          </div>

          {/* 1. Guardian Identifier */}
          <div className="form-group-field">
            <label htmlFor="consent-guardian-id" className="form-field-label">
              Guardian / Caregiver Identifier <span className="req-asterisk">*</span>
            </label>
            <input
              id="consent-guardian-id"
              type="text"
              className="alias-input"
              value={guardianId}
              onChange={(e) => setGuardianId(e.target.value)}
              placeholder="e.g. caregiver_primary or guardian_jane"
              disabled={isSubmitting}
            />
            <span className="field-hint">Enter your authorized caregiver or legal guardian identifier.</span>
          </div>

          {/* 2. Guardian Explicit Consent */}
          <div className="checkbox-consent-card">
            <label className="checkbox-label-row">
              <input
                type="checkbox"
                id="guardian-consent-check"
                checked={guardianConsent}
                onChange={(e) => setGuardianConsent(e.target.checked)}
                disabled={isSubmitting}
              />
              <span className="checkbox-text-group">
                <strong>I certify that guardian / caregiver consent is granted <span className="req-asterisk">*</span></strong>
                <span className="checkbox-subtext">
                  Authorizes location tracking, mobility risk analysis, and temporary companion invitations for {userId}.
                </span>
              </span>
            </label>
          </div>

          {/* 3. Patient Legal/Cognitive Capacity Assessment */}
          <div className="capacity-assessment-card">
            <span className="field-section-title">
              Patient Cognitive &amp; Legal Capacity Assessment <span className="req-asterisk">*</span>
            </span>
            <p className="capacity-desc">
              Does <strong>{userId}</strong> retain legal and cognitive capacity to provide informed consent for mobility monitoring?
            </p>
            <div className="radio-options-row">
              <label className="radio-option-label">
                <input
                  type="radio"
                  name="end-user-capacity"
                  value="yes"
                  checked={endUserApplicable === true}
                  onChange={() => setEndUserApplicable(true)}
                  disabled={isSubmitting}
                />
                <span><strong>Yes</strong> — Patient retains capacity (Requires patient consent)</span>
              </label>
              <label className="radio-option-label">
                <input
                  type="radio"
                  name="end-user-capacity"
                  value="no"
                  checked={endUserApplicable === false}
                  onChange={() => setEndUserApplicable(false)}
                  disabled={isSubmitting}
                />
                <span><strong>No</strong> — Guardian acts on patient&apos;s behalf (Clinical/Caregiver assessment)</span>
              </label>
            </div>
          </div>

          {/* 4. End-User Consent Checkbox (Conditional) */}
          {endUserApplicable === true && (
            <div className="checkbox-consent-card patient-consent-row">
              <label className="checkbox-label-row">
                <input
                  type="checkbox"
                  id="patient-consent-check"
                  checked={endUserConsent}
                  onChange={(e) => setEndUserConsent(e.target.checked)}
                  disabled={isSubmitting}
                />
                <span className="checkbox-text-group">
                  <strong>Patient has granted explicit consent <span className="req-asterisk">*</span></strong>
                  <span className="checkbox-subtext">
                    The patient has been informed and explicitly consents to sharing location telemetry.
                  </span>
                </span>
              </label>
            </div>
          )}

          {/* 5. Data Retention Acknowledgment */}
          <div className="checkbox-consent-card">
            <label className="checkbox-label-row">
              <input
                type="checkbox"
                id="retention-ack-check"
                checked={dataRetentionAck}
                onChange={(e) => setDataRetentionAck(e.target.checked)}
                disabled={isSubmitting}
              />
              <span className="checkbox-text-group">
                <strong>Data Retention Policy Acknowledgment <span className="req-asterisk">*</span></strong>
                <span className="checkbox-subtext">
                  Acknowledge that raw GPS coordinates are subject to automated 30-day retention policies and H3 spatial aggregation.
                </span>
              </span>
            </label>
          </div>

          {/* Actions */}
          <div className="consent-buttons-row">
            <button
              type="submit"
              className="companion-btn companion-btn-primary consent-accept-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Recording Consent...' : '✓ Record Authorized Patient Consent'}
            </button>
            <button
              type="button"
              className="companion-btn companion-btn-secondary"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
