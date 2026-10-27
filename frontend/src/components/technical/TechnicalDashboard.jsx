import React, { useState, useEffect } from 'react'
import MapPanel from '../MapPanel'
import RiskPanel from '../RiskPanel'
import AlertPanel from '../AlertPanel'

export default function TechnicalDashboard({
  userId,
  setUserId,
  isConnected,
  handleConnectClick,
  handleDisconnectClick,
  handleReconnectClick,
  riskWsStatus,
  alertWsStatus,
  currentLocation,
  locationHistory,
  trailMetadata = [],
  activeRisk,
  latestRisk,
  historyEntries,
  alertMessages,
  handleAlertUpdate,
  historyLoading,
  historyError,
  fetchHistory,
  patientStatus = null,
  latestWearable = null,
  safeZones = [],
  refreshSafeZones = null,
  alertsLoading = false,
  alertsError = null,
  fetchAlerts = null,
}) {
  const [inputUserId, setInputUserId] = useState(userId)
  const [activeCompanion, setActiveCompanion] = useState(null)
  const [patientConsent, setPatientConsent] = useState(null)
  const [companionLoading, setCompanionLoading] = useState(false)
  const [showAssociateForm, setShowAssociateForm] = useState(false)
  const [companionInputId, setCompanionInputId] = useState('')
  const [companionRelationship, setCompanionRelationship] = useState('family')
  const [isAssociating, setIsAssociating] = useState(false)
  const [associationError, setAssociationError] = useState(null)
  const [simStep, setSimStep] = useState(0)
  const [isSimulating, setIsSimulating] = useState(false)
  const [simFeedback, setSimFeedback] = useState(null)
  const [elevatedHrSim, setElevatedHrSim] = useState(false)

  const handleSimulateCompanionStep = async (scenario) => {
    if (!userId || !userId.trim()) return
    setIsSimulating(true)
    setSimFeedback(null)
    const nextStep = simStep + 1
    setSimStep(nextStep)

    try {
      const payload = {
        patient_id: userId.trim(),
        scenario: scenario,
        step_number: nextStep,
        companion_id: activeCompanion?.companion_user_id || 'comp_alex',
        companion_alias: activeCompanion?.companion_alias || 'Family Guide (Alex)',
        heart_rate_bpm: elevatedHrSim ? 105 : 75,
      }
      const res = await fetch('/api/companion/simulation/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setSimFeedback({
        scenario,
        status: data.display_status,
        distance: data.distance_meters,
        action: data.action,
        alertId: data.alert_id,
        tier: data.alert_tier,
      })
      await fetchActiveCompanion(userId)
      if (fetchAlerts) {
        await fetchAlerts(userId)
      }
    } catch (err) {
      console.error('Simulation step error:', err)
      setSimFeedback({ error: err.message })
    } finally {
      setIsSimulating(false)
    }
  }

  useEffect(() => {
    setInputUserId(userId)
  }, [userId])

  const handleSelectPatient = (targetId) => {
    const trimmed = (targetId || '').trim()
    if (!trimmed) return
    setInputUserId(trimmed)
    if (isConnected) {
      handleDisconnectClick()
    }
    setUserId(trimmed)
  }

  const handleFormSubmit = (e) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    const trimmed = (inputUserId || '').trim()
    if (!trimmed) return
    if (!isConnected || trimmed !== userId) {
      if (isConnected && trimmed !== userId) {
        handleDisconnectClick()
      }
      if (trimmed !== userId) {
        setUserId(trimmed)
      }
      handleConnectClick(e, trimmed)
    } else {
      handleDisconnectClick()
    }
  }

  // Check authorized consent status for current patient
  const fetchConsent = async (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) {
      setPatientConsent(null)
      return
    }
    try {
      const res = await fetch(`/api/consent/${encodeURIComponent(targetUserId.trim())}`)
      if (res.ok) {
        const data = await res.json()
        if (data.status === 'not_found') {
          setPatientConsent({ is_consented: false, record: null })
        } else if (data.guardian_consent?.granted) {
          const isFull = data.end_user_consent_applicable ? Boolean(data.end_user_consent?.granted) : true
          setPatientConsent({ is_consented: isFull, record: data })
        } else {
          setPatientConsent({ is_consented: false, record: data })
        }
      } else {
        setPatientConsent({ is_consented: false, reason: 'Consent record missing' })
      }
    } catch (err) {
      setPatientConsent({ is_consented: false, reason: 'Error checking consent' })
    }
  }

  // Query actual active companion / family association for selected patient
  const fetchActiveCompanion = async (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) {
      setActiveCompanion(null)
      return
    }

    try {
      let url = `/api/companion/sessions/active/${encodeURIComponent(targetUserId.trim())}`
      if (currentLocation?.lat != null && currentLocation?.lng != null) {
        url += `?patient_lat=${currentLocation.lat}&patient_lng=${currentLocation.lng}`
      }
      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setActiveCompanion(data.active_session)
      } else {
        setActiveCompanion(null)
      }
    } catch (err) {
      console.warn('Failed to fetch active companion for technical inspector:', err)
      setActiveCompanion(null)
    }
  }

  // Poll consent and active companion on mount, userId change, and periodically during monitoring
  useEffect(() => {
    fetchConsent(userId)
    fetchActiveCompanion(userId)
    const interval = setInterval(() => {
      fetchConsent(userId)
      fetchActiveCompanion(userId)
    }, 2500)
    return () => clearInterval(interval)
  }, [userId, currentLocation?.lat, currentLocation?.lng])

  // Helper to determine and display missing prerequisites when accompaniment is unverified
  const getMissingPrerequisite = () => {
    if (patientConsent && !patientConsent.is_consented) {
      return `Patient consent required: ${userId || 'This patient'} has no authorized consent record on file. Patient consent must be recorded before companion or family accompaniment can be authorized.`
    }
    if (!activeCompanion || activeCompanion.status !== 'ACTIVE') {
      return `No authorized family member or active Trusted Companion is currently associated with ${userId || 'this patient'}. The map displays the patient only.`
    }
    if (activeCompanion.last_companion_lat == null || activeCompanion.last_companion_lng == null) {
      return `Companion location fix is unavailable. Awaiting device telemetry.`
    }
    if (activeCompanion.is_location_stale) {
      return `Companion location telemetry fix is stale (>60s old). Fresh telemetry is required to verify accompaniment.`
    }
    if (currentLocation?.lat == null || currentLocation?.lng == null) {
      return `Patient location telemetry is unavailable. Replay or stream patient telemetry to verify accompaniment.`
    }
    return null
  }

  // Explicitly associate family member or companion
  const handleCreateAssociation = async (e) => {
    e.preventDefault()
    if (!userId || !userId.trim() || !companionInputId.trim()) return

    setIsAssociating(true)
    setAssociationError(null)

    try {
      const isFamily = companionRelationship === 'family'
      const res = await fetch('/api/companion/sessions/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: userId.trim(),
          caregiver_id: 'caregiver_primary',
          validity_minutes: 1440, // 24 hours
          companion_id: companionInputId.trim(),
          relationship: companionRelationship,
          companion_alias: isFamily
            ? `Family Member (${companionInputId.trim()})`
            : `Trusted Companion (${companionInputId.trim()})`,
        }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Server returned HTTP ${res.status}`)
      }

      setCompanionInputId('')
      setShowAssociateForm(false)
      await fetchActiveCompanion(userId)
    } catch (err) {
      console.error('Failed to create companion association:', err)
      setAssociationError(err.message)
    } finally {
      setIsAssociating(false)
    }
  }

  // End active companion session / association
  const handleEndCompanionSession = async () => {
    if (!activeCompanion?.session_id) return

    setIsAssociating(true)
    try {
      const res = await fetch(`/api/companion/sessions/${activeCompanion.session_id}/end`, {
        method: 'POST',
      })
      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`)
      }
      setActiveCompanion(null)
    } catch (err) {
      console.error('Failed to end companion session:', err)
    } finally {
      setIsAssociating(false)
    }
  }

  return (
    <div className="technical-dashboard-container">
      <header className="header">
        <h1>Predictive Geofencing Operator Dashboard</h1>
        <p>Stage 8: Live Spatial Visualization, TreeSHAP Explainability &amp; Quarantine Suppression</p>
      </header>

      {/* Connection & Controls Bar */}
      <div className="controls-card">
        <form onSubmit={handleFormSubmit} className="form-group">
          <label htmlFor="user-id-input">Patient ID:</label>
          <input
            id="user-id-input"
            type="text"
            className="form-input"
            value={inputUserId}
            onChange={(e) => setInputUserId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                handleFormSubmit(e)
              }
            }}
            placeholder="Enter any patient ID..."
          />
          {patientStatus && (
            <span
              className="patient-registry-status"
              style={{
                fontSize: '12px',
                padding: '4px 8px',
                borderRadius: '4px',
                fontWeight: 600,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                backgroundColor: patientStatus.exists ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
                color: patientStatus.exists ? '#059669' : '#dc2626',
                border: `1px solid ${patientStatus.exists ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
              }}
            >
              {patientStatus.exists
                ? (historyEntries?.length > 0
                    ? `✓ Verified Patient (${historyEntries.length} records)`
                    : `⚠️ Patient Verified (0 location records)`)
                : `❓ Unknown Patient ID (Not Found)`}
            </span>
          )}
          {!isConnected || inputUserId.trim() !== userId ? (
            <button type="submit" className="btn btn-primary">
              Connect
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-danger"
              onClick={handleDisconnectClick}
            >
              Disconnect
            </button>
          )}
          {isConnected && (riskWsStatus === 'closed' || riskWsStatus === 'error' || alertWsStatus === 'closed' || alertWsStatus === 'error') && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleReconnectClick}
            >
              Reconnect
            </button>
          )}
        </form>

        <div className="status-indicators">
          <span className={`status-pill status-${riskWsStatus}`}>
            Risk WS: {riskWsStatus.toUpperCase()}
          </span>
          <span className={`status-pill status-${alertWsStatus}`}>
            Alerts WS: {alertWsStatus.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Real Associated Family / Companion Accompaniment Strip (Requirement 1-5) */}
      <div className="technical-accompaniment-strip" role="region" aria-label="Family and Companion Accompaniment">
        <div className="tech-acc-header-row">
          <div className="tech-acc-title-group">
            <span className="tech-acc-icon">👥</span>
            <strong>Associated Family &amp; Companion Tracking</strong>
            <span className="tech-acc-patient-tag">
              Patient ID: <strong>{userId || 'None'}</strong>
            </span>
          </div>

          <div className="tech-acc-status-badges">
            {activeCompanion && activeCompanion.status === 'ACTIVE' ? (
              <span
                className={`tech-acc-status-pill ${
                  activeCompanion.accompaniment_status === 'With Companion'
                    ? 'pill-accompanied'
                    : activeCompanion.accompaniment_status === 'Separated'
                    ? 'pill-separated'
                    : 'pill-unverified'
                }`}
              >
                <span className="pill-dot" />
                {activeCompanion.accompaniment_status || 'Accompaniment Unverified'}
                {activeCompanion.separation_distance_m != null
                  ? ` (${Math.round(activeCompanion.separation_distance_m)}m)`
                  : ''}
              </span>
            ) : (
              <span className="tech-acc-status-pill pill-unverified">
                <span className="pill-dot" />
                Accompaniment Unverified
              </span>
            )}

            {activeCompanion?.is_location_stale && (
              <span className="tech-acc-stale-tag">⚠️ Stale Fix (&gt;60s)</span>
            )}
            {activeCompanion && !activeCompanion.is_location_stale && activeCompanion.last_companion_lat != null && (
              <span className="tech-acc-live-tag">🟢 Fresh Live Fix</span>
            )}
          </div>
        </div>

        {/* Details and Actions Row */}
        {activeCompanion && activeCompanion.status === 'ACTIVE' ? (
          <div className="tech-acc-body-card">
            {activeCompanion.accompaniment_status === 'Accompaniment Unverified' && (
              <div className="tech-acc-prereq-banner" role="status">
                <span className="prereq-icon">ℹ️</span>
                <div className="prereq-content">
                  <strong>Accompaniment Unverified — Missing Prerequisite:</strong>
                  <p className="prereq-text">{getMissingPrerequisite()}</p>
                </div>
              </div>
            )}

            <div className="tech-acc-details-grid">
              <div className="tech-acc-detail-item">
                <span className="detail-label">Associated Person:</span>
                <span className="detail-value mono">
                  {activeCompanion.companion_user_id
                    ? `${activeCompanion.companion_user_id} (${activeCompanion.relationship === 'family' ? 'Family Member' : 'Companion'})`
                    : activeCompanion.companion_alias || 'Trusted Companion'}
                </span>
              </div>
              <div className="tech-acc-detail-item">
                <span className="detail-label">Actual Location:</span>
                <span className="detail-value mono">
                  {activeCompanion.last_companion_lat != null && activeCompanion.last_companion_lng != null
                    ? `${activeCompanion.last_companion_lat.toFixed(5)}, ${activeCompanion.last_companion_lng.toFixed(5)}`
                    : 'Awaiting device fix'}
                </span>
              </div>
              <div className="tech-acc-detail-item">
                <span className="detail-label">Telemetry Timestamp:</span>
                <span className="detail-value mono">
                  {activeCompanion.last_companion_timestamp
                    ? new Date(activeCompanion.last_companion_timestamp).toLocaleTimeString()
                    : '—'}
                </span>
              </div>
              <div className="tech-acc-detail-item">
                <span className="detail-label">Distance to Patient:</span>
                <span className="detail-value">
                  {activeCompanion.separation_distance_m != null
                    ? `${Math.round(activeCompanion.separation_distance_m)}m (Threshold: 50m)`
                    : 'Distance uncalculated'}
                </span>
              </div>
              <div className="tech-acc-detail-item">
                <span className="detail-label">Configured Safe Zone:</span>
                <span className="detail-value">
                  {activeCompanion.safe_zone?.configured
                    ? `${activeCompanion.safe_zone.name} (${activeCompanion.safe_zone.radius_m ? `${activeCompanion.safe_zone.radius_m}m radius` : 'custom'})`
                    : `None configured for ${userId || 'this patient'}`}
                </span>
              </div>
            </div>

            {activeCompanion.safe_zone_departure_accompanied && activeCompanion.safe_zone?.configured && (
              <div className="tech-acc-safe-departure-note">
                🛡️ Patient and companion are outside their configured safe zone ({activeCompanion.safe_zone.name}) together (Accompanied — no family separation).
              </div>
            )}

            <div className="tech-acc-sim-card" style={{ marginTop: '14px', padding: '12px 14px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#1e293b' }}>
                  🧪 Companion Walking Simulation (Development Test)
                </span>
                <span style={{ fontSize: '11px', background: 'rgba(30, 41, 59, 0.08)', padding: '2px 6px', borderRadius: '4px', color: '#475569' }}>
                  [Simulated Companion Data]
                </span>
              </div>
              <p style={{ fontSize: '12px', color: '#64748b', margin: '0 0 10px 0' }}>
                Test multi-person companion walking alongside <strong>{userId}</strong> across three distinct test scenarios:
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={isSimulating}
                  onClick={() => handleSimulateCompanionStep('WALKING_TOGETHER')}
                  title="Simulate companion walking side-by-side (<10m)"
                >
                  🚶‍♂️🚶‍♀️ Walk Together (&lt;10m)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={isSimulating}
                  onClick={() => handleSimulateCompanionStep('SEPARATING')}
                  title="Simulate companion separating beyond 50m (step 1 -> Pending, step 2 -> Separated)"
                >
                  ↗️ Separate (&gt;50m)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={isSimulating}
                  onClick={() => handleSimulateCompanionStep('REJOINING')}
                  title="Simulate companion rejoining patient (<10m auto-resolves alert)"
                >
                  🤝 Rejoin (&lt;10m)
                </button>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '12px', color: '#334155', cursor: 'pointer', marginLeft: '6px' }}>
                  <input
                    type="checkbox"
                    checked={elevatedHrSim}
                    onChange={(e) => setElevatedHrSim(e.target.checked)}
                  />
                  <span>Simulate HR Arousal (105 BPM)</span>
                </label>
              </div>

              {simFeedback && (
                <div style={{ marginTop: '10px', fontSize: '12px', padding: '6px 10px', borderRadius: '6px', background: simFeedback.error ? '#fef2f2' : '#f0fdf4', color: simFeedback.error ? '#dc2626' : '#166534', border: `1px solid ${simFeedback.error ? '#fecaca' : '#bbf7d0'}` }}>
                  {simFeedback.error ? (
                    <span>⚠️ {simFeedback.error}</span>
                  ) : (
                    <span>
                      Result: Status <strong>{simFeedback.status}</strong> | Distance: <strong>{simFeedback.distance ? `${Math.round(simFeedback.distance)}m` : '—'}</strong>
                      {simFeedback.action !== 'NONE' && ` | Action: ${simFeedback.action}`}
                      {simFeedback.alertId && ` (Alert ID: ${simFeedback.alertId.slice(0, 8)}... Tier ${simFeedback.tier})`}
                    </span>
                  )}
                </div>
              )}
            </div>

            <div className="tech-acc-actions-row" style={{ marginTop: '12px' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleEndCompanionSession}
                disabled={isAssociating}
              >
                Disconnect Association / End Session
              </button>
            </div>
          </div>
        ) : (
          <div className="tech-acc-idle-card">
            <div className="tech-acc-prereq-banner" role="status">
              <span className="prereq-icon">ℹ️</span>
              <div className="prereq-content">
                <strong>Accompaniment Unverified — Missing Prerequisite:</strong>
                <p className="prereq-text">{getMissingPrerequisite()}</p>
              </div>
            </div>

            <p className="tech-acc-idle-desc">
              No authorized family member or active Trusted Companion is currently associated with{' '}
              <strong>{userId || 'this patient'}</strong>. The map displays the patient only.
            </p>

            {patientConsent && !patientConsent.is_consented ? (
              <div className="tech-acc-consent-alert" role="alert">
                🔒 <strong>Patient Consent Required:</strong> Cannot associate a family member or companion without authorized patient consent on file for <strong>{userId}</strong>.
              </div>
            ) : !showAssociateForm ? (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setShowAssociateForm(true)}
                disabled={!userId}
              >
                🔗 Associate Family Member or Companion
              </button>
            ) : (
              <form onSubmit={handleCreateAssociation} className="tech-acc-form">
                <div className="tech-acc-form-row">
                  <div className="tech-acc-input-group">
                    <label htmlFor="tech-companion-id">Companion User ID / Alias:</label>
                    <input
                      id="tech-companion-id"
                      type="text"
                      className="form-input"
                      value={companionInputId}
                      onChange={(e) => setCompanionInputId(e.target.value)}
                      placeholder="e.g. test_user_3"
                      disabled={isAssociating}
                    />
                  </div>
                  <div className="tech-acc-input-group">
                    <label htmlFor="tech-relationship">Relationship:</label>
                    <select
                      id="tech-relationship"
                      className="form-input"
                      value={companionRelationship}
                      onChange={(e) => setCompanionRelationship(e.target.value)}
                      disabled={isAssociating}
                    >
                      <option value="family">Family Member</option>
                      <option value="trusted_companion">Trusted Companion</option>
                    </select>
                  </div>
                  <div className="tech-acc-form-buttons">
                    <button
                      type="submit"
                      className="btn btn-primary btn-sm"
                      disabled={isAssociating || !companionInputId.trim()}
                    >
                      {isAssociating ? 'Associating...' : `Link to ${userId}`}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setShowAssociateForm(false)
                        setAssociationError(null)
                      }}
                      disabled={isAssociating}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
                {associationError && (
                  <div className="tech-acc-error" role="alert">
                    ⚠️ {associationError}
                  </div>
                )}
              </form>
            )}
          </div>
        )}
      </div>

      {/* Main Dashboard Grid: Map on Left (60%), Risk & Alerts on Right (40%) */}
      <div className="dashboard-main-grid">
        <div className="map-column">
          <MapPanel
            currentLocation={currentLocation}
            locationHistory={locationHistory}
            trailMetadata={trailMetadata}
            currentRiskTier={activeRisk?.risk_tier}
            currentRiskScore={activeRisk?.risk_score}
            userId={userId}
            activeCompanion={activeCompanion}
            configuredSafeZone={activeCompanion?.safe_zone}
            patientSafeZones={safeZones}
            patientStatus={patientStatus}
          />
        </div>

        <div className="side-column">
          <RiskPanel
            currentRisk={latestRisk}
            historyEntries={historyEntries}
            alertMessages={alertMessages}
            isConnected={isConnected}
            trailMetadata={trailMetadata}
            userId={userId}
            latestWearable={latestWearable}
          />

          <AlertPanel
            alerts={alertMessages}
            onAlertUpdate={handleAlertUpdate}
            userId={userId}
            isLoading={alertsLoading}
            error={alertsError}
          />
        </div>
      </div>

      {/* Bottom Panel: Historical Risk Records (REST) */}
      <div className="panel" style={{ marginTop: '24px' }}>
        <div className="panel-header">
          <h2 className="panel-title">
            Historical Risk Records (REST: GET /api/risk-history/{userId}?limit=100)
          </h2>
          {isConnected && (
            <button
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: '12px' }}
              onClick={() => fetchHistory(userId)}
            >
              Refresh
            </button>
          )}
        </div>

        {historyLoading && <div className="empty-state">Loading risk history...</div>}
        {historyError && (
          <div className="empty-state" style={{ color: '#b91c1c' }}>
            Failed to load risk history: {historyError}
          </div>
        )}

        {!historyLoading && !historyError && historyEntries.length === 0 && (
          <div className="empty-state">
            {patientStatus && !patientStatus.exists
              ? `Unknown patient ID "${userId}": Patient not found in the database.`
              : `No recorded location data for this patient ("${userId}").`}
          </div>
        )}

        {!historyLoading && !historyError && historyEntries.length > 0 && (() => {
          const hasAnyMlEval = historyEntries.some((e) => e.ml_evaluation != null)
          const sortedEntries = [...historyEntries].sort(
            (a, b) => new Date(b.timestamp) - new Date(a.timestamp) || b.id - a.id
          )
          return (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Risk Tier</th>
                    <th>Risk Score</th>
                    <th>Polling Tier</th>
                    <th>Grid Cell ID</th>
                    {hasAnyMlEval && <th>ML Eval (Shadow)</th>}
                  </tr>
                </thead>
                <tbody>
                  {sortedEntries.map((entry) => (
                    <tr key={entry.id}>
                      <td className="mono">{entry.id}</td>
                      <td className="mono">{entry.timestamp}</td>
                      <td>
                        <span className={`tier-badge tier-${entry.risk_tier}`}>
                          {entry.risk_tier}
                        </span>
                      </td>
                      <td className="mono">{Number(entry.risk_score).toFixed(1)}</td>
                      <td className="mono">Tier {entry.polling_tier}</td>
                      <td className="mono">{entry.grid_cell_id || '-'}</td>
                      {hasAnyMlEval && (
                        <td className="mono">
                          {entry.ml_evaluation?.calibrated_probability != null
                            ? `${(entry.ml_evaluation.calibrated_probability * 100).toFixed(1)}%`
                            : entry.ml_evaluation?.evidence_status === 'INSUFFICIENT_EVIDENCE'
                            ? 'Collecting'
                            : entry.ml_evaluation?.evidence_status === 'ERROR'
                            ? 'Unavailable'
                            : '-'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
