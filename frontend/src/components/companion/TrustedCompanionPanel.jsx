import React, { useState, useEffect } from 'react'
import {
  evaluateAccompaniment,
  ACCOMPANIMENT_STATUS,
} from './companionLogic.js'


export default function TrustedCompanionPanel({
  userId,
  currentLocation,
  onCompanionStateChange,
  patientConsent,
  onOpenConsentModal,
}) {
  const [activeSession, setActiveSession] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [copiedLink, setCopiedLink] = useState(false)
  const [testSimulating, setTestSimulating] = useState(false)
  const [now, setNow] = useState(Date.now())

  // Keep timestamp age fresh
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Poll active session for selected patient
  const fetchActiveSession = async () => {
    if (!userId || !userId.trim()) {
      setActiveSession(null)
      return
    }

    try {
      const pLat = currentLocation?.lat
      const pLng = currentLocation?.lng
      let url = `/api/companion/sessions/active/${encodeURIComponent(userId.trim())}`
      if (pLat != null && pLng != null) {
        url += `?patient_lat=${pLat}&patient_lng=${pLng}`
      }

      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setActiveSession(data.active_session)
        if (onCompanionStateChange) {
          onCompanionStateChange(data.active_session)
        }
      }
    } catch (err) {
      console.warn('Could not poll companion session:', err)
    }
  }

  // Poll on mount and when userId or currentLocation changes
  useEffect(() => {
    fetchActiveSession()
    const pollTimer = setInterval(fetchActiveSession, 3000)
    return () => clearInterval(pollTimer)
  }, [userId, currentLocation?.lat, currentLocation?.lng])

  // Create temporary session
  const handleCreateSession = async () => {
    if (!userId || !userId.trim()) return
    setIsLoading(true)
    setErrorMsg(null)

    try {
      const res = await fetch('/api/companion/sessions/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: userId.trim(),
          caregiver_id: 'caregiver_primary',
          validity_minutes: 60,
        }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Server returned ${res.status}`)
      }

      await fetchActiveSession()
    } catch (err) {
      console.error('Failed to create companion session:', err)
      setErrorMsg(`Failed to start session: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // End session
  const handleEndSession = async () => {
    if (!activeSession?.session_id) return
    setIsLoading(true)
    setErrorMsg(null)

    try {
      const res = await fetch(`/api/companion/sessions/${activeSession.session_id}/end`, {
        method: 'POST',
      })
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`)
      }
      setActiveSession(null)
      if (onCompanionStateChange) {
        onCompanionStateChange(null)
      }
    } catch (err) {
      console.error('Failed to end companion session:', err)
      setErrorMsg(`Failed to end session: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // Build full invite URL
  const inviteToken = activeSession?.invite_token
  const inviteUrl = inviteToken
    ? `${window.location.origin}/?companion_invite=${encodeURIComponent(inviteToken)}`
    : ''

  const handleCopyLink = () => {
    if (!inviteUrl) return
    navigator.clipboard?.writeText(inviteUrl).then(() => {
      setCopiedLink(true)
      setTimeout(() => setCopiedLink(false), 2500)
    })
  }

  // Evaluate accompaniment state locally with fresh timer
  const accompaniment = evaluateAccompaniment({
    patientLocation: currentLocation,
    companionLocation:
      activeSession?.last_companion_lat != null && activeSession?.last_companion_lng != null
        ? { lat: activeSession.last_companion_lat, lng: activeSession.last_companion_lng }
        : null,
    companionTimestamp: activeSession?.last_companion_timestamp,
    thresholdMeters: 50,
    maxStaleSeconds: 60,
    currentTime: now,
  })

  // Format expiration countdown
  const getExpirationText = (isoStr) => {
    if (!isoStr) return ''
    const diffSec = Math.floor((new Date(isoStr).getTime() - now) / 1000)
    if (diffSec <= 0) return 'Expired'
    const mins = Math.floor(diffSec / 60)
    const secs = diffSec % 60
    return `${mins}m ${secs}s remaining`
  }

  // Test simulation helper: transmit real timestamped coordinates as consenting companion
  const handleTransmitTestLocation = async (offsetMeters = 15) => {
    if (!activeSession?.session_id || !inviteToken) return
    setTestSimulating(true)
    try {
      // First ensure consent is recorded if still pending
      if (!activeSession.consent_granted) {
        await fetch(`/api/companion/invite/${inviteToken}/consent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ granted: true, companion_alias: 'Trusted Companion' }),
        })
      }

      const pLat = currentLocation?.lat || 37.7749
      const pLng = currentLocation?.lng || -122.4194
      // Offset slightly to test distance
      const latOffset = (offsetMeters / 111111.0)
      const compLat = pLat + latOffset
      const compLng = pLng

      await fetch(`/api/companion/sessions/${activeSession.session_id}/location`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          companion_token: inviteToken,
          lat: compLat,
          lng: compLng,
          accuracy_m: 6.0,
          timestamp: new Date().toISOString(),
        }),
      })

      await fetchActiveSession()
    } catch (err) {
      console.error('Failed to transmit companion test location:', err)
    } finally {
      setTestSimulating(false)
    }
  }

  return (
    <div className="panel trusted-companion-card" role="region" aria-label="Trusted Companion Mode">
      <div className="panel-header">
        <div className="companion-panel-title-group">
          <h2 className="panel-title">👥 Trusted Companion Mode</h2>
          <span className="badge-count">
            {activeSession
              ? activeSession.status === 'ACTIVE'
                ? 'Active Session'
                : 'Pending Invite'
              : 'Standby'}
          </span>
        </div>
      </div>

      <div className="trusted-companion-body">
        {errorMsg && (
          <div className="companion-error-banner" role="alert">
            <div className="companion-error-message-group">
              <span>⚠️ {errorMsg}</span>
              {errorMsg.toLowerCase().includes('consent') && onOpenConsentModal && (
                <button
                  type="button"
                  className="companion-btn-inline-setup"
                  onClick={onOpenConsentModal}
                >
                  🛡️ Set Up Patient Consent Now
                </button>
              )}
            </div>
            <button type="button" className="banner-dismiss-btn" onClick={() => setErrorMsg(null)}>×</button>
          </div>
        )}

        {/* State A: No Active Session */}
        {!activeSession && (
          <div className="companion-idle-state">
            <div className="companion-idle-icon-disc">🤝</div>
            <div className="companion-idle-text">
              <h4 className="companion-idle-title">Temporary Trusted Companion</h4>
              <p className="companion-idle-desc">
                Invite a trusted friend to accompany <strong>User ({userId || 'selected patient'})</strong> for a temporary walking session using a secure, expiring invitation link.
              </p>

              {patientConsent && !patientConsent.isConsented && (
                <div className="companion-consent-notice-box" role="alert">
                  <div className="consent-notice-content">
                    <span className="consent-notice-icon">⚠️</span>
                    <div className="consent-notice-text">
                      <strong>Patient Consent Required (§2.7)</strong>
                      <p>
                        Authorized consent must be recorded before inviting a companion for {userId || 'this patient'}.
                      </p>
                    </div>
                  </div>
                  {onOpenConsentModal && (
                    <button
                      type="button"
                      className="companion-btn companion-btn-secondary consent-setup-btn"
                      onClick={onOpenConsentModal}
                    >
                      🛡️ Set Up Patient Consent
                    </button>
                  )}
                </div>
              )}

              <div className="companion-privacy-bullets">
                <span className="privacy-pill">✓ Temporary & Expiring</span>
                <span className="privacy-pill">✓ Explicit Location Consent</span>
                <span className="privacy-pill">✓ Personal ID Protected</span>
                <span className="privacy-pill">✓ Companion Can Stop Anytime</span>
              </div>
            </div>
            <button
              type="button"
              className="companion-btn companion-btn-primary"
              onClick={handleCreateSession}
              disabled={isLoading || !userId}
            >
              {isLoading ? 'Generating Invitation...' : '➕ Start Companion Session & Invite Friend'}
            </button>
          </div>
        )}

        {/* State B: Session Pending Companion Consent */}
        {activeSession && activeSession.status === 'PENDING_INVITE' && (
          <div className="companion-pending-state">
            <div className="pending-badge-row">
              <span className="companion-status-tag tag-pending">
                ⏳ Awaiting Friend&apos;s Consent
              </span>
              <span className="countdown-pill tabular-nums">
                ⏱️ {getExpirationText(activeSession.expires_at)}
              </span>
            </div>

            <p className="pending-instructions">
              Send this secure, single-use invitation link to the companion. They must open it on their device and explicitly consent to temporary location sharing before accompaniment tracking begins.
            </p>

            <div className="invite-link-box">
              <input
                type="text"
                readOnly
                value={inviteUrl}
                className="invite-link-input"
                aria-label="Expiring Companion Invitation Link"
              />
              <button
                type="button"
                className="companion-btn companion-btn-secondary"
                onClick={handleCopyLink}
              >
                {copiedLink ? '✓ Copied!' : '📋 Copy Link'}
              </button>
            </div>

            <div className="pending-actions-row">
              <button
                type="button"
                className="companion-btn companion-btn-sim"
                onClick={() => handleTransmitTestLocation(15)}
                disabled={testSimulating}
                title="Simulate companion accepting invitation and sending live GPS fix near patient"
              >
                {testSimulating ? 'Simulating...' : '🧪 Simulate Companion Joining (Dev Test)'}
              </button>
              <button
                type="button"
                className="companion-btn companion-btn-danger"
                onClick={handleEndSession}
                disabled={isLoading}
              >
                Cancel Session
              </button>
            </div>
          </div>
        )}

        {/* State C: Active Accompaniment Session */}
        {activeSession && activeSession.status === 'ACTIVE' && (
          <div className="companion-active-state">
            <div className="active-header-row">
              <div className="active-companion-identity">
                <span className="companion-avatar-badge">👤</span>
                <div className="companion-name-group">
                  <span className="companion-alias-title">{activeSession.companion_alias || 'Trusted Companion'}</span>
                  <span className="companion-privacy-tag">Temporary Guest · Personal ID Protected</span>
                </div>
              </div>

              <div className="accompaniment-status-pill-group">
                <span
                  className={`accompaniment-badge ${
                    accompaniment.status === ACCOMPANIMENT_STATUS.WITH_COMPANION
                      ? 'badge-with-companion'
                      : accompaniment.status === ACCOMPANIMENT_STATUS.SEPARATED
                      ? 'badge-separated'
                      : 'badge-unverified'
                  }`}
                >
                  <span className="badge-dot" />
                  {accompaniment.status}
                </span>
              </div>
            </div>

            {/* Metrics & Separation Row */}
            <div className="companion-metrics-grid">
              <div className="companion-metric-item">
                <span className="c-metric-eyebrow">Distance to Patient</span>
                <span className="c-metric-value tabular-nums">
                  {accompaniment.distanceMeters != null
                    ? `${Math.round(accompaniment.distanceMeters)}m`
                    : '—'}
                </span>
                <span className="c-metric-hint">Threshold: 50m</span>
              </div>

              <div className="companion-metric-item">
                <span className="c-metric-eyebrow">Companion Telemetry</span>
                <span className="c-metric-value">
                  {accompaniment.isStale ? '⚠️ Stale' : '🟢 Live Fix'}
                </span>
                <span className="c-metric-hint tabular-nums">
                  {accompaniment.ageSeconds != null ? `${accompaniment.ageSeconds}s ago` : 'Awaiting fix'}
                </span>
              </div>

              <div className="companion-metric-item">
                <span className="c-metric-eyebrow">Session Expiration</span>
                <span className="c-metric-value tabular-nums">
                  {getExpirationText(activeSession.expires_at)}
                </span>
                <span className="c-metric-hint">Auto-concludes</span>
              </div>
            </div>

            {/* Accompaniment Narrative & Safe Zone Notice */}
            <div
              className={`companion-narrative-card ${
                accompaniment.safeZoneDepartureAccompanied ? 'narrative-safe-departure' : ''
              }`}
            >
              <div className="narrative-icon">
                {accompaniment.status === ACCOMPANIMENT_STATUS.WITH_COMPANION
                  ? '🤝'
                  : accompaniment.status === ACCOMPANIMENT_STATUS.SEPARATED
                  ? '⚠️'
                  : '📡'}
              </div>
              <div className="narrative-text">
                <strong>{accompaniment.statusMessage}</strong>
                {accompaniment.safeZoneDepartureAccompanied && (
                  <div className="safe-zone-accompaniment-hint">
                    Safe zone boundary departed together with trusted companion. Group separation alert suppressed while accompanied.
                  </div>
                )}
              </div>
            </div>

            {/* Active Companion Actions */}
            <div className="active-actions-row">
              <button
                type="button"
                className="companion-btn companion-btn-sim-quick"
                onClick={() => handleTransmitTestLocation(10)}
                disabled={testSimulating}
                title="Send updated location fix near patient"
              >
                📍 Send GPS Fix Near Patient (10m)
              </button>
              <button
                type="button"
                className="companion-btn companion-btn-sim-quick"
                onClick={() => handleTransmitTestLocation(120)}
                disabled={testSimulating}
                title="Send updated location fix separated from patient"
              >
                🚶 Send Separated Fix (120m)
              </button>
              <button
                type="button"
                className="companion-btn companion-btn-danger"
                onClick={handleEndSession}
                disabled={isLoading}
              >
                End Companion Session
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
