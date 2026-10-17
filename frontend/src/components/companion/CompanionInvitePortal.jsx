import React, { useState, useEffect, useRef } from 'react'

export default function CompanionInvitePortal({ inviteToken, onClose }) {
  const [inviteData, setInviteData] = useState(null)
  const [sessionStatus, setSessionStatus] = useState('LOADING') // LOADING, INVITE, ACTIVE, DECLINED, ENDED, ERROR
  const [errorMsg, setErrorMsg] = useState(null)
  const [alias, setAlias] = useState('Trusted Companion')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastTransmitted, setLastTransmitted] = useState(null)
  const [gpsError, setGpsError] = useState(null)
  const [gpsAccuracy, setGpsAccuracy] = useState(null)
  const [now, setNow] = useState(Date.now())

  const watchIdRef = useRef(null)

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Fetch invitation details
  useEffect(() => {
    if (!inviteToken) return

    const loadInvite = async () => {
      try {
        const res = await fetch(`/api/companion/invite/${encodeURIComponent(inviteToken)}`)
        if (!res.ok) {
          const data = await res.json().catch(() => ({}))
          throw new Error(data.detail || `Server returned ${res.status}`)
        }
        const data = await res.json()
        setInviteData(data)
        if (data.status === 'ACTIVE') {
          setSessionStatus('ACTIVE')
        } else {
          setSessionStatus('INVITE')
        }
      } catch (err) {
        console.error('Failed to load invite details:', err)
        setErrorMsg(err.message)
        setSessionStatus('ERROR')
      }
    }

    loadInvite()
  }, [inviteToken])

  // Stop geolocation watcher on unmount or session end
  useEffect(() => {
    return () => {
      if (watchIdRef.current != null && navigator.geolocation) {
        navigator.geolocation.clearWatch(watchIdRef.current)
        watchIdRef.current = null
      }
    }
  }, [])

  // Start real browser geolocation watching once ACTIVE
  const startGeolocation = (sessionId) => {
    if (!navigator.geolocation) {
      setGpsError('Geolocation is not supported by your browser.')
      return
    }

    setGpsError(null)
    const handleSuccess = async (pos) => {
      const lat = pos.coords.latitude
      const lng = pos.coords.longitude
      const acc = pos.coords.accuracy

      setGpsAccuracy(acc)
      try {
        const res = await fetch(`/api/companion/sessions/${sessionId}/location`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            companion_token: inviteToken,
            lat,
            lng,
            accuracy_m: acc,
            timestamp: new Date().toISOString(),
          }),
        })
        if (res.ok) {
          setLastTransmitted(new Date().toLocaleTimeString())
        }
      } catch (err) {
        console.warn('Failed to transmit companion GPS position:', err)
      }
    }

    const handleError = (err) => {
      console.warn('Geolocation error:', err)
      if (err.code === 1) {
        setGpsError('Location permission denied by user. Enable browser location to share updates.')
      } else {
        setGpsError(`Device GPS error (${err.message}).`)
      }
    }

    watchIdRef.current = navigator.geolocation.watchPosition(handleSuccess, handleError, {
      enableHighAccuracy: true,
      maximumAge: 10000,
      timeout: 15000,
    })
  }

  // Handle consent submission
  const handleConsent = async (granted) => {
    if (!inviteToken) return
    setIsSubmitting(true)
    setErrorMsg(null)

    try {
      const res = await fetch(`/api/companion/invite/${encodeURIComponent(inviteToken)}/consent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          granted,
          companion_alias: alias || 'Trusted Companion',
        }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Consent request failed (${res.status})`)
      }

      const data = await res.json()
      if (!granted) {
        setSessionStatus('DECLINED')
      } else {
        setSessionStatus('ACTIVE')
        startGeolocation(data.session_id)
      }
    } catch (err) {
      console.error('Failed to record consent:', err)
      setErrorMsg(err.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  // Companion stops sharing
  const handleStopSharing = async () => {
    if (!inviteData?.session_id || !inviteToken) return
    setIsSubmitting(true)

    try {
      if (watchIdRef.current != null && navigator.geolocation) {
        navigator.geolocation.clearWatch(watchIdRef.current)
        watchIdRef.current = null
      }

      await fetch(
        `/api/companion/sessions/${inviteData.session_id}/stop-sharing?companion_token=${encodeURIComponent(inviteToken)}`,
        { method: 'POST' }
      )
      setSessionStatus('ENDED')
    } catch (err) {
      console.error('Failed to stop sharing:', err)
    } finally {
      setIsSubmitting(false)
    }
  }

  // Developer simulation helper for manual GPS transmission
  const handleManualTestTransmission = async () => {
    if (!inviteData?.session_id || !inviteToken) return
    try {
      // Default coordinate in neighborhood (San Francisco reference)
      const lat = 37.7749 + (Math.random() - 0.5) * 0.0004
      const lng = -122.4194 + (Math.random() - 0.5) * 0.0004
      await fetch(`/api/companion/sessions/${inviteData.session_id}/location`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          companion_token: inviteToken,
          lat,
          lng,
          accuracy_m: 8.0,
          timestamp: new Date().toISOString(),
        }),
      })
      setLastTransmitted(new Date().toLocaleTimeString())
    } catch (err) {
      console.error('Manual transmission failed:', err)
    }
  }

  return (
    <div className="companion-portal-overlay" role="dialog" aria-modal="true" aria-label="Trusted Companion Invitation">
      <div className="companion-portal-modal">
        <div className="companion-portal-header">
          <div className="portal-header-icon">🤝</div>
          <div className="portal-header-text">
            <h2 className="portal-title">Trusted Companion Invitation</h2>
            <span className="portal-subtitle">Temporary Walking Accompaniment Session</span>
          </div>
          {onClose && (
            <button type="button" className="portal-close-btn" onClick={onClose} aria-label="Close portal">
              ×
            </button>
          )}
        </div>

        <div className="companion-portal-body">
          {errorMsg && (
            <div className="portal-error-banner" role="alert">
              ⚠️ {errorMsg}
            </div>
          )}

          {/* State 1: Loading */}
          {sessionStatus === 'LOADING' && (
            <div className="portal-loading-box">
              <span className="portal-spinner">🔄</span>
              <span>Validating invitation token and session status...</span>
            </div>
          )}

          {/* State 2: Error or Expired */}
          {sessionStatus === 'ERROR' && (
            <div className="portal-state-box state-error">
              <span className="state-icon">⚠️</span>
              <h3 className="state-title">Invitation Unavailable</h3>
              <p className="state-desc">
                {errorMsg || 'This invitation link is invalid, has expired, or the session has already concluded.'}
              </p>
              {onClose && (
                <button type="button" className="companion-btn companion-btn-secondary" onClick={onClose}>
                  Return to Dashboard
                </button>
              )}
            </div>
          )}

          {/* State 3: Explicit Consent Review */}
          {sessionStatus === 'INVITE' && (
            <div className="portal-consent-box">
              <div className="consent-callout">
                <span className="callout-kicker">INVITATION DETAILS</span>
                <p className="callout-patient">
                  You are invited to accompany <strong>User ({inviteData?.patient_id || 'Patient'})</strong>.
                </p>
                <p className="callout-disclosure">
                  {inviteData?.privacy_disclosure ||
                    'You are invited as a temporary trusted companion for this walking session. You do not become a permanent family member, and your personal ID will never be exposed on the caregiver dashboard.'}
                </p>
              </div>

              <div className="consent-guarantees">
                <div className="guarantee-row">
                  <span className="guarantee-icon">🛡️</span>
                  <div className="guarantee-text">
                    <strong>Privacy-Preserving Alias</strong>
                    <span>Your personal identity and account ID are never revealed on the caregiver dashboard.</span>
                  </div>
                </div>
                <div className="guarantee-row">
                  <span className="guarantee-icon">⏱️</span>
                  <div className="guarantee-text">
                    <strong>Temporary & Automatically Expiring</strong>
                    <span>Location updates terminate automatically when the session concludes.</span>
                  </div>
                </div>
                <div className="guarantee-row">
                  <span className="guarantee-icon">✋</span>
                  <div className="guarantee-text">
                    <strong>Full Consent Control</strong>
                    <span>You can click &ldquo;Stop Sharing&rdquo; at any time to immediately revoke location sharing.</span>
                  </div>
                </div>
              </div>

              <div className="companion-alias-input-group">
                <label htmlFor="companion-alias-input" className="alias-label">
                  Your Display Alias (e.g. &ldquo;Friend Sarah&rdquo; or &ldquo;Companion&rdquo;):
                </label>
                <input
                  id="companion-alias-input"
                  type="text"
                  className="alias-input"
                  value={alias}
                  onChange={(e) => setAlias(e.target.value)}
                  placeholder="Trusted Companion"
                  maxLength={30}
                />
              </div>

              <div className="consent-buttons-row">
                <button
                  type="button"
                  className="companion-btn companion-btn-primary consent-accept-btn"
                  onClick={() => handleConsent(true)}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Recording Consent...' : '✓ I Explicitly Consent & Share My Location'}
                </button>
                <button
                  type="button"
                  className="companion-btn companion-btn-secondary"
                  onClick={() => handleConsent(false)}
                  disabled={isSubmitting}
                >
                  Decline Invitation
                </button>
              </div>
            </div>
          )}

          {/* State 4: Active Session with Real Location Sharing */}
          {sessionStatus === 'ACTIVE' && (
            <div className="portal-active-box">
              <div className="portal-active-banner">
                <span className="active-dot-ping" />
                <span className="active-banner-text">
                  Location Sharing Active · Accompaniment Linked
                </span>
              </div>

              <div className="active-telemetry-status-card">
                <div className="status-metric-row">
                  <span className="status-metric-label">GPS Tracking Status:</span>
                  <span className="status-metric-val">
                    {gpsError ? '⚠️ GPS Notice' : '🟢 Transmitting Live Updates'}
                  </span>
                </div>
                {gpsAccuracy != null && (
                  <div className="status-metric-row">
                    <span className="status-metric-label">Device Accuracy:</span>
                    <span className="status-metric-val mono">±{Math.round(gpsAccuracy)}m</span>
                  </div>
                )}
                {lastTransmitted && (
                  <div className="status-metric-row">
                    <span className="status-metric-label">Last Location Transmitted:</span>
                    <span className="status-metric-val mono">{lastTransmitted}</span>
                  </div>
                )}
                {gpsError && (
                  <div className="gps-error-note">
                    {gpsError}
                  </div>
                )}
              </div>

              <p className="active-companion-help">
                Keep this tab open while accompanying the monitored person. Your device will report updates so the caregiver can verify you are safely travelling together.
              </p>

              <div className="active-controls-cluster">
                <button
                  type="button"
                  className="companion-btn companion-btn-sim-quick"
                  onClick={handleManualTestTransmission}
                  title="Transmit an immediate location fix"
                >
                  📡 Transmit Current GPS Fix
                </button>
                <button
                  type="button"
                  className="companion-btn companion-btn-danger"
                  onClick={handleStopSharing}
                  disabled={isSubmitting}
                >
                  ✋ Stop Sharing & Leave Session
                </button>
              </div>
            </div>
          )}

          {/* State 5: Declined or Ended */}
          {(sessionStatus === 'DECLINED' || sessionStatus === 'ENDED') && (
            <div className="portal-state-box state-ended">
              <span className="state-icon">✋</span>
              <h3 className="state-title">
                {sessionStatus === 'DECLINED' ? 'Invitation Declined' : 'Location Sharing Stopped'}
              </h3>
              <p className="state-desc">
                {sessionStatus === 'DECLINED'
                  ? 'You declined the companion invitation. No location data was shared.'
                  : 'You have stopped sharing your location. This companion session is terminated.'}
              </p>
              {onClose && (
                <button type="button" className="companion-btn companion-btn-primary" onClick={onClose}>
                  Close Window
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
