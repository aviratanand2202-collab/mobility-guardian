import React, { useState, useEffect } from 'react'
import MapPanel from '../MapPanel'
import TrustedCompanionPanel from '../companion/TrustedCompanionPanel'
import PatientConsentModal from './PatientConsentModal'

function getRelativeTime(timestamp, currentTime) {
  if (!timestamp) return '-'
  try {
    const diffMs = currentTime - new Date(timestamp).getTime()
    const diffSec = Math.max(0, Math.floor(diffMs / 1000))
    if (diffSec < 60) return `${diffSec}s ago`
    const diffMin = Math.floor(diffSec / 60)
    if (diffMin < 60) return `${diffMin}m ago`
    const diffHours = Math.floor(diffMin / 60)
    return `${diffHours}h ago`
  } catch {
    return timestamp
  }
}

export default function CaregiverDashboard({
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
  historyEntries = [],
  alertMessages,
  handleAlertUpdate,
  alertsLoading = false,
  alertsError = null,
  fetchAlerts,
  patientStatus = null,
  onSwitchToTechnical,
}) {
  const [inputUserId, setInputUserId] = useState(userId)
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    setInputUserId(userId)
  }, [userId])
  const [mapActionTrigger, setMapActionTrigger] = useState(null)
  const [dismissingIds, setDismissingIds] = useState(new Set())
  const [dismissErrors, setDismissErrors] = useState({})
  const [recalibrationPrompt, setRecalibrationPrompt] = useState(null)
  const [expandedFeatureIndices, setExpandedFeatureIndices] = useState(new Set())
  const [expandedAlertIds, setExpandedAlertIds] = useState(new Set())
  const [activeCompanion, setActiveCompanion] = useState(null)
  const [patientConsent, setPatientConsent] = useState(null)
  const [consentLoading, setConsentLoading] = useState(false)
  const [showConsentModal, setShowConsentModal] = useState(false)

  // Fetch patient consent record per §2.7
  const fetchPatientConsent = async (targetId) => {
    if (!targetId || !targetId.trim()) {
      setPatientConsent(null)
      return
    }
    setConsentLoading(true)
    try {
      const res = await fetch(`/api/consent/${encodeURIComponent(targetId.trim())}`)
      if (res.ok) {
        const data = await res.json()
        if (data.status === 'not_found') {
          setPatientConsent({ isConsented: false, record: null })
        } else if (data.guardian_consent?.granted) {
          setPatientConsent({ isConsented: true, record: data })
        } else {
          setPatientConsent({ isConsented: false, record: data })
        }
      } else {
        setPatientConsent({ isConsented: false, record: null })
      }
    } catch (err) {
      console.warn('Failed to fetch patient consent:', err)
      setPatientConsent({ isConsented: false, record: null })
    } finally {
      setConsentLoading(false)
    }
  }

  useEffect(() => {
    fetchPatientConsent(userId)
  }, [userId])

  const handleConsentRecorded = (record) => {
    setPatientConsent({ isConsented: true, record })
    fetchPatientConsent(userId)
  }

  const toggleFeatureExpand = (idx) => {
    setExpandedFeatureIndices((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) {
        next.delete(idx)
      } else {
        next.add(idx)
      }
      return next
    })
  }

  const toggleAlertExpand = (alertId) => {
    setExpandedAlertIds((prev) => {
      const next = new Set(prev)
      if (next.has(alertId)) {
        next.delete(alertId)
      } else {
        next.add(alertId)
      }
      return next
    })
  }

  // Keep relative times fresh
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [])

  // Dynamic user label (never hard-code Eleanor or Arthur)
  const personLabel = userId && userId.trim() ? `User (${userId.trim()})` : 'Monitored Person'

  // 1. Connection-State Semantics per Refinement 2
  const isRiskOnline = isConnected && riskWsStatus === 'open'
  const isRiskConnecting = isConnected && riskWsStatus === 'connecting'
  const isRiskOffline = !isConnected || riskWsStatus === 'closed' || riskWsStatus === 'error' || riskWsStatus === 'disconnected'

  const isAlertOnline = isConnected && alertWsStatus === 'open'
  const isAlertConnecting = isConnected && alertWsStatus === 'connecting'
  const isAlertOffline = !isConnected || alertWsStatus === 'closed' || alertWsStatus === 'error' || alertWsStatus === 'disconnected'

  // An alert-socket issue must NOT mark location monitoring as stopped if risk telemetry is flowing!
  const isAlertIssue = isRiskOnline && (alertWsStatus === 'closed' || alertWsStatus === 'error')

  // Data availability check
  const hasLiveTelemetry = isRiskOnline && latestRisk != null
  const hasHistoricalOnly = isRiskOnline && latestRisk == null && historyEntries && historyEntries.length > 0
  const isAwaitingTelemetry = isRiskOnline && latestRisk == null && (!historyEntries || historyEntries.length === 0)

  // Telemetry freshness & accuracy
  const effectiveRisk = latestRisk || activeRisk
  const telemetryTimestamp = effectiveRisk?.timestamp
  const telemetryTimeMs = telemetryTimestamp ? new Date(telemetryTimestamp).getTime() : null
  const telemetryAgeSec =
    telemetryTimeMs && !isNaN(telemetryTimeMs)
      ? Math.max(0, Math.floor((now - telemetryTimeMs) / 1000))
      : null
  const isTelemetryStale = telemetryAgeSec != null && telemetryAgeSec > 45
  const isLiveTelemetryFresh = isRiskOnline && latestRisk != null && !isTelemetryStale
  const telemetryAge = telemetryTimestamp ? getRelativeTime(telemetryTimestamp, now) : null

  const hasValidLocation = Boolean(
    (currentLocation && currentLocation.lat != null && currentLocation.lng != null) ||
    (locationHistory && locationHistory.length > 0)
  )

  const gpsAccuracy = effectiveRisk?.sensor_metrics?.horizontal_accuracy_m
  const batteryPct = effectiveRisk?.sensor_metrics?.battery_pct
  const isBatteryAvailable = isRiskOnline && batteryPct != null

  const isLowBattery =
    effectiveRisk?.battery_override_active ||
    effectiveRisk?.polling_instruction?.mode === 'PULSE' ||
    effectiveRisk?.polling_instruction?.mode === 'LAST_GASP' ||
    (batteryPct != null && batteryPct < 15)

  let batteryStatusText = 'Battery: Unavailable'
  let batteryPillClass = 'pill-inactive'
  if (isBatteryAvailable) {
    if (isLowBattery) {
      batteryStatusText = `⚠️ Battery Low (${batteryPct}%)`
      batteryPillClass = 'pill-warning'
    } else if (isTelemetryStale) {
      batteryStatusText = `🔋 Battery ${batteryPct}% (Stale)`
      batteryPillClass = 'pill-warning'
    } else if (hasHistoricalOnly) {
      batteryStatusText = `🔋 Battery ${batteryPct}% (Cached)`
      batteryPillClass = 'pill-subtle'
    } else {
      batteryStatusText = `🔋 Battery ${batteryPct}%`
      batteryPillClass = 'pill-normal'
    }
  } else if (!isRiskOnline) {
    batteryStatusText = 'Battery: Stream Offline'
    batteryPillClass = 'pill-inactive'
  }

  // Truthful status mapping (evidence-based only, no unconditional proof of safety)
  let statusTitle = 'Stationary / Low Activity'
  let statusDescription = 'Low motion observed in recent observation window. Note: stationary state does not verify overall safety.'
  let statusBadgeClass = 'caregiver-badge-quiescent'
  let statusCardClass = 'caregiver-hero-quiescent'
  let statusIcon = '🛡️'
  let qualitativeNotice = null
  let actionTitle = 'STABLE / STANDBY'
  let actionContent = 'Routine check or standby. Device reports low activity.'
  let isActionUrgent = false

  const riskTier = effectiveRisk?.risk_tier || 'QUIESCENT'
  const isQuiescent = riskTier === 'QUIESCENT'
  const isNormalTransit = riskTier === 'NORMAL_TRANSIT'
  const isSuspicious = riskTier === 'SUSPICIOUS'
  const isCritical = riskTier === 'CRITICAL'

  if (isRiskOffline) {
    statusTitle = 'Monitoring Disconnected'
    statusDescription = 'Live telemetry stream is disconnected. Real-time safety status cannot be verified.'
    statusBadgeClass = 'caregiver-badge-offline'
    statusCardClass = 'caregiver-hero-offline'
    statusIcon = '⚪'
    actionTitle = 'RECONNECT REQUIRED'
    actionContent = 'Stream is disconnected. Reconnect stream or check device connectivity.'
    isActionUrgent = true
  } else if (isRiskConnecting) {
    statusTitle = 'Connecting to Stream'
    statusDescription = 'Establishing real-time connection with monitoring service...'
    statusBadgeClass = 'caregiver-badge-connecting'
    statusCardClass = 'caregiver-hero-connecting'
    statusIcon = '🔄'
    actionTitle = 'CONNECTING'
    actionContent = 'Connecting... Live telemetry will appear momentarily.'
    isActionUrgent = false
  } else if (isAwaitingTelemetry) {
    statusTitle = 'Awaiting Initial Telemetry'
    statusDescription = 'Connected to monitoring service. Waiting for first device location and sensor fix.'
    statusBadgeClass = 'caregiver-badge-waiting'
    statusCardClass = 'caregiver-hero-waiting'
    statusIcon = '📡'
    actionTitle = 'AWAITING FIX'
    actionContent = 'Ensure monitored device is powered on and transmitting.'
    isActionUrgent = false
  } else if (hasHistoricalOnly) {
    const lastRecordedTime = effectiveRisk?.timestamp ? getRelativeTime(effectiveRisk.timestamp, now) : 'previously'
    if (isQuiescent) {
      statusTitle = 'Last Known: Stationary'
      statusDescription = `Stationary or low activity when last reported (${lastRecordedTime}). Awaiting live telemetry fix.`
      statusBadgeClass = 'caregiver-badge-quiescent'
      statusCardClass = 'caregiver-hero-quiescent'
      statusIcon = '🛡️'
      actionTitle = 'HISTORICAL FIX'
      actionContent = 'Last reported status was stationary. Awaiting live telemetry.'
    } else if (isNormalTransit) {
      statusTitle = 'Last Known: Normal Pace'
      statusDescription = `Moving at a steady pace when last reported (${lastRecordedTime}). Awaiting live telemetry fix.`
      statusBadgeClass = 'caregiver-badge-normal'
      statusCardClass = 'caregiver-hero-normal'
      statusIcon = '🚶'
      actionTitle = 'HISTORICAL FIX'
      actionContent = 'Movement was steady when last reported. Awaiting live telemetry.'
    } else if (isSuspicious) {
      statusTitle = 'Last Known: Unusual Movement'
      statusDescription = `Unusual movement was flagged when last reported (${lastRecordedTime}). Awaiting live telemetry fix.`
      statusBadgeClass = 'caregiver-badge-suspicious'
      statusCardClass = 'caregiver-hero-suspicious'
      statusIcon = '⚠️'
      qualitativeNotice = 'Unusual movement pattern detected; monitoring closely.'
      actionTitle = 'VERIFY STATUS'
      actionContent = `Verify status of ${personLabel}. Last report was unusual.`
      isActionUrgent = true
    } else if (isCritical) {
      statusTitle = 'Last Known: Attention Recommended'
      statusDescription = `High-risk movement was flagged when last reported (${lastRecordedTime}). Awaiting live telemetry fix.`
      statusBadgeClass = 'caregiver-badge-critical'
      statusCardClass = 'caregiver-hero-critical'
      statusIcon = '🚨'
      qualitativeNotice = 'High-risk movement pattern detected. A prompt check is recommended.'
      actionTitle = 'VERIFY STATUS'
      actionContent = `Prompt verification recommended. Check on ${personLabel} directly.`
      isActionUrgent = true
    }
  } else if (isTelemetryStale) {
    const staleRelativeTime = telemetryTimestamp ? getRelativeTime(telemetryTimestamp, now) : 'previously'
    if (isQuiescent) {
      statusTitle = 'Last Known: Stationary'
      statusDescription = `Stationary or low activity when last reported (${staleRelativeTime}). Live telemetry is delayed or stale.`
      statusBadgeClass = 'caregiver-badge-quiescent'
      statusCardClass = 'caregiver-hero-quiescent'
      statusIcon = '🛡️'
      actionTitle = 'STANDBY / STALE GPS'
      actionContent = `Telemetry delayed (${staleRelativeTime}). Standing by for updated telemetry fix.`
      isActionUrgent = false
    } else if (isNormalTransit) {
      statusTitle = 'Last Known: Normal Pace'
      statusDescription = `Moving at a steady pace when last reported (${staleRelativeTime}). Live telemetry is delayed or stale.`
      statusBadgeClass = 'caregiver-badge-normal'
      statusCardClass = 'caregiver-hero-normal'
      statusIcon = '🚶'
      actionTitle = 'STANDBY / STALE GPS'
      actionContent = `Telemetry delayed (${staleRelativeTime}). Standing by for updated telemetry fix.`
      isActionUrgent = false
    } else if (isSuspicious) {
      statusTitle = 'Last Known: Unusual Movement'
      statusDescription = `Unusual movement was flagged when last reported (${staleRelativeTime}). Telemetry is stale; verify status.`
      statusBadgeClass = 'caregiver-badge-suspicious'
      statusCardClass = 'caregiver-hero-suspicious'
      statusIcon = '⚠️'
      qualitativeNotice = 'Unusual movement pattern detected; monitoring closely.'
      actionTitle = 'VERIFY STATUS'
      actionContent = `Verify status of ${personLabel}. Last report was unusual and telemetry is delayed.`
      isActionUrgent = true
    } else if (isCritical) {
      statusTitle = 'Last Known: Attention Recommended'
      statusDescription = `High-risk movement was flagged when last reported (${staleRelativeTime}). Telemetry is stale; prompt check recommended.`
      statusBadgeClass = 'caregiver-badge-critical'
      statusCardClass = 'caregiver-hero-critical'
      statusIcon = '🚨'
      qualitativeNotice = 'High-risk movement pattern detected. A prompt check is recommended.'
      actionTitle = 'VERIFY STATUS'
      actionContent = `Prompt verification recommended. Check on ${personLabel} directly.`
      isActionUrgent = true
    }
  } else {
    // Live telemetry actively streaming
    if (isQuiescent) {
      statusTitle = 'Stationary / Low Activity'
      statusDescription = 'Low motion observed in recent window. Note: stationary state does not verify overall safety.'
      statusBadgeClass = 'caregiver-badge-quiescent'
      statusCardClass = 'caregiver-hero-quiescent'
      statusIcon = '🛡️'
      actionTitle = 'STABLE / STANDBY'
      actionContent = 'Routine check or standby. Device reports low activity.'
    } else if (isNormalTransit) {
      statusTitle = 'Normal Movement Pace'
      statusDescription = 'Moving at a steady, consistent pace in the latest observation window.'
      statusBadgeClass = 'caregiver-badge-normal'
      statusCardClass = 'caregiver-hero-normal'
      statusIcon = '🚶'
      actionTitle = 'STABLE / TRANSIT'
      actionContent = 'Routine monitoring active. Steady pace observed.'
    } else if (isSuspicious) {
      statusTitle = 'Unusual Movement Pattern'
      statusDescription = 'Irregular motion, pacing, or elevated velocity variance detected. Keep aware.'
      statusBadgeClass = 'caregiver-badge-suspicious'
      statusCardClass = 'caregiver-hero-suspicious'
      statusIcon = '⚠️'
      qualitativeNotice = 'Unusual movement pattern detected; monitoring closely.'
      actionTitle = 'ATTENTION RECOMMENDED'
      actionContent = `Check on ${personLabel} directly or confirm their location.`
      isActionUrgent = true
    } else if (isCritical) {
      statusTitle = 'Attention Recommended'
      statusDescription = 'Rapid progression or high-risk movement pattern detected. Immediate check suggested.'
      statusBadgeClass = 'caregiver-badge-critical'
      statusCardClass = 'caregiver-hero-critical'
      statusIcon = '🚨'
      qualitativeNotice = 'High-risk movement pattern detected. A prompt check is recommended.'
      actionTitle = 'IMMEDIATE CHECK'
      actionContent = `Prompt verification recommended. Check on ${personLabel} immediately.`
      isActionUrgent = true
    }
  }

  // Separate active (PENDING) and historical/dismissed (DISMISSED_SAFE) alert counts
  const pendingAlerts = alertMessages.filter((a) => a.status === 'PENDING')
  const pendingCount = pendingAlerts.length
  const dismissedCount = alertMessages.filter((a) => a.status === 'DISMISSED_SAFE').length

  // Truthful Alert Empty-State Presentation per Safety Correction
  let emptyAlertTitle = 'No Active Alerts Recorded'
  let emptyAlertDesc = 'No geofence breach or safety alert events have been triggered.'
  let emptyAlertPillText = 'No active alerts'
  let emptyAlertZoneText = 'Continuous monitoring'
  let emptyAlertCardClass = 'empty-alert-nominal'
  let emptyAlertDiscClass = 'disc-nominal'
  let emptyAlertIcon = '🛡️'

  if (alertsLoading && alertMessages.length === 0) {
    emptyAlertTitle = 'Syncing Safety Alerts...'
    emptyAlertDesc = 'Checking server for active or past safety alerts...'
    emptyAlertPillText = 'Syncing'
    emptyAlertZoneText = 'Querying records'
    emptyAlertCardClass = 'empty-alert-awaiting'
    emptyAlertDiscClass = 'disc-awaiting'
    emptyAlertIcon = '🔄'
  } else if (alertsError && alertMessages.length === 0) {
    emptyAlertTitle = 'Alert Records Unavailable'
    emptyAlertDesc = `Unable to verify existing safety alerts: ${alertsError}. Live alerts will still stream if triggered.`
    emptyAlertPillText = 'Sync Error'
    emptyAlertZoneText = 'Safeguards unverified'
    emptyAlertCardClass = 'empty-alert-critical'
    emptyAlertDiscClass = 'disc-critical'
    emptyAlertIcon = '⚠️'
  } else if (isAlertOffline) {
    emptyAlertTitle = isRiskOffline ? 'Alerts Unavailable While Disconnected' : 'Alert Monitoring Unavailable'
    emptyAlertDesc = isRiskOffline
      ? 'Reconnect monitoring to retrieve saved alerts. Real-time safety status cannot be verified while disconnected.'
      : 'The alert stream is disconnected while movement telemetry is flowing. Critical safety notifications are currently unavailable.'
    emptyAlertPillText = 'Stream Offline'
    emptyAlertZoneText = 'Alerts inactive'
    emptyAlertCardClass = 'empty-alert-offline'
    emptyAlertDiscClass = 'disc-offline'
    emptyAlertIcon = '⚪'
  } else if (isAlertConnecting) {
    emptyAlertTitle = 'Connecting to Alert Service'
    emptyAlertDesc = 'Establishing real-time connection with alert monitoring stream...'
    emptyAlertPillText = 'Connecting'
    emptyAlertZoneText = 'Handshake in progress'
    emptyAlertCardClass = 'empty-alert-awaiting'
    emptyAlertDiscClass = 'disc-awaiting'
    emptyAlertIcon = '🔄'
  } else if (isCritical) {
    emptyAlertTitle = 'No Alert Triggered — High Movement Risk'
    emptyAlertDesc = hasLiveTelemetry
      ? 'Movement algorithms flag high-risk motion (Tier 3), but an alert notification has not met the multi-window threshold or boundary breach trigger yet. Check on the monitored person directly.'
      : 'Last recorded movement was flagged as high risk (Tier 3), but no formal alert event is on file. Awaiting live telemetry fix.'
    emptyAlertPillText = 'High Risk Movement'
    emptyAlertZoneText = 'Check person directly'
    emptyAlertCardClass = 'empty-alert-critical'
    emptyAlertDiscClass = 'disc-critical'
    emptyAlertIcon = '🚨'
  } else if (isSuspicious) {
    emptyAlertTitle = 'No Alert Triggered — Unusual Movement'
    emptyAlertDesc = hasLiveTelemetry
      ? 'Movement algorithms flag unusual motion (Tier 2), but an alert notification has not met the multi-window trigger threshold yet. Continue monitoring closely.'
      : 'Last recorded movement was flagged as unusual (Tier 2), but no formal alert event is on file. Awaiting live telemetry fix.'
    emptyAlertPillText = 'Elevated Risk'
    emptyAlertZoneText = 'Monitoring closely'
    emptyAlertCardClass = 'empty-alert-suspicious'
    emptyAlertDiscClass = 'disc-suspicious'
    emptyAlertIcon = '⚠️'
  } else if (!hasLiveTelemetry || isTelemetryStale) {
    // Stale, awaiting initial telemetry, or historical only
    emptyAlertTitle = 'No Active Alerts Recorded'
    emptyAlertDesc = hasHistoricalOnly
      ? 'No alert events currently recorded. Note: live telemetry is not actively streaming, so current safety cannot be inferred from the lack of alerts.'
      : isTelemetryStale
      ? `No alert events currently recorded. Note: telemetry is delayed (${telemetryAge || 'stale'}), so current safety cannot be inferred from the lack of alerts.`
      : 'Connected to monitoring service. Awaiting device telemetry to evaluate real-time motion and zone boundaries.'
    emptyAlertPillText = hasHistoricalOnly ? 'Historical Fix Only' : isTelemetryStale ? 'Telemetry Stale' : 'Awaiting Telemetry'
    emptyAlertZoneText = 'Safety unverified'
    emptyAlertCardClass = 'empty-alert-awaiting'
    emptyAlertDiscClass = 'disc-awaiting'
    emptyAlertIcon = '📡'
  } else {
    // Live telemetry streaming, QUIESCENT or NORMAL_TRANSIT
    emptyAlertTitle = 'No Active Alerts Recorded'
    emptyAlertDesc = 'No boundary breaches or multi-window movement alerts have been triggered in the current window. Note: absence of alerts does not independently verify overall safety.'
    emptyAlertPillText = 'No active alerts'
    emptyAlertZoneText = 'Continuous monitoring'
    emptyAlertCardClass = 'empty-alert-nominal'
    emptyAlertDiscClass = 'disc-nominal'
    emptyAlertIcon = '🛡️'
  }

  // Explainability top features in plain language (no raw SHAP decimals)
  const topFeatures = activeRisk?.explainability?.top_features || []

  // Dismissal flow (uses exact same backend API & optimistic update)
  const handleDismiss = async (alert) => {
    const alertId = alert.alert_id
    if (!alertId || dismissingIds.has(alertId)) return

    setDismissErrors((prev) => {
      const next = { ...prev }
      delete next[alertId]
      return next
    })

    const previousStatus = alert.status
    const previousMultiplier = alert.sensitivity_multiplier

    setDismissingIds((prev) => new Set(prev).add(alertId))

    // Optimistic UI update
    handleAlertUpdate(alertId, {
      status: 'DISMISSED_SAFE',
      isOptimistic: true,
    })

    try {
      const res = await fetch(`/api/alerts/${alertId}/dismiss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }

      const data = await res.json()

      // Case A: 4th dismissal requiring formal recalibration
      if (data.status === 'recalibration_required') {
        handleAlertUpdate(alertId, {
          status: previousStatus,
          isOptimistic: false,
        })
        setRecalibrationPrompt({
          alertId,
          gridCellId: alert.grid_cell_id,
          message:
            data.message ||
            'Routine change detected. Add this route to the Safe Mobility Profile permanently?',
        })
        return
      }

      // Case B: Normal dismissal
      handleAlertUpdate(alertId, {
        status: 'DISMISSED_SAFE',
        sensitivity_multiplier: data.cell_sensitivity_multiplier,
        dismissal_count: data.dismissal_count_in_window,
        isOptimistic: false,
      })
    } catch (err) {
      console.error(`Failed to dismiss alert ${alertId}:`, err)
      handleAlertUpdate(alertId, {
        status: previousStatus,
        sensitivity_multiplier: previousMultiplier,
        isOptimistic: false,
      })
      setDismissErrors((prev) => ({
        ...prev,
        [alertId]: `Unable to confirm and dismiss alert: ${err.message}`,
      }))
    } finally {
      setDismissingIds((prev) => {
        const next = new Set(prev)
        next.delete(alertId)
        return next
      })
    }
  }

  const handleRecalibrationAnswer = (choice) => {
    if (choice === 'Yes' || choice === 'Update Baseline') {
      console.info(
        `[Safe Mobility Profile] Caregiver requested baseline sensitivity update for cell ${recalibrationPrompt?.gridCellId}. Note: baseline adjustments modify alert frequency and do not certify route safety.`
      )
    } else {
      console.info(
        `[Safe Mobility Profile] Caregiver retained standard sensitivity safeguards for cell ${recalibrationPrompt?.gridCellId}.`
      )
    }
    setRecalibrationPrompt(null)
  }

  return (
    <div className="caregiver-dashboard-container">
      {/* 1. Serene Hearth Header with Integrated Person Identifier & Controls */}
      <header className="caregiver-header">
        <div className="caregiver-header-main">
          <div className="caregiver-identity-lockup">
            <div className="caregiver-avatar">
              <span className="avatar-icon">🛡️</span>
            </div>
            <div className="caregiver-identity-info">
              <div className="caregiver-eyebrow">CAREGIVER SAFETY INTEL</div>
              <div className="caregiver-title-row">
                <h1 className="caregiver-title">Mobility Guardian</h1>
                <div className="caregiver-person-badge">
                  <span className={`beacon-dot ${isRiskOnline ? 'beacon-live' : 'beacon-offline'}`} />
                  <span>Monitored: <strong>{personLabel}</strong></span>
                  <span className="badge-divider">·</span>
                  <span className="beacon-status-text">{isRiskOnline ? 'Beacon Linked' : 'Stream Disconnected'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Integrated Monitoring Controls Form */}
          <form
            onSubmit={(e) => {
              if (e) {
                e.preventDefault()
                e.stopPropagation()
              }
              const trimmed = (inputUserId || '').trim()
              if (!trimmed) return
              if (!isConnected) {
                if (trimmed !== userId) {
                  setUserId(trimmed)
                }
                handleConnectClick(e, trimmed)
              } else {
                handleDisconnectClick()
              }
            }}
            className="caregiver-controls-inline"
          >
            <div className="control-group-compact">
              <label htmlFor="caregiver-user-id" className="control-label-compact">Person ID</label>
              <input
                id="caregiver-user-id"
                type="text"
                className="control-input-compact"
                value={inputUserId}
                onChange={(e) => setInputUserId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    const trimmed = (inputUserId || '').trim()
                    if (!trimmed) return
                    if (isConnected) handleDisconnectClick()
                    setUserId(trimmed)
                  }
                }}
                placeholder="e.g. test_user_000"
              />
            </div>
            {!isConnected ? (
              <button type="submit" className="caregiver-btn caregiver-btn-primary">
                Start Monitoring
              </button>
            ) : (
              <button
                type="button"
                className="caregiver-btn caregiver-btn-danger"
                onClick={handleDisconnectClick}
              >
                Stop Monitoring
              </button>
            )}
            {isConnected &&
              (riskWsStatus === 'closed' ||
                riskWsStatus === 'error' ||
                alertWsStatus === 'closed' ||
                alertWsStatus === 'error') && (
                <button
                  type="button"
                  className="caregiver-btn caregiver-btn-secondary"
                  onClick={handleReconnectClick}
                >
                  Reconnect
                </button>
              )}
          </form>
        </div>

        {/* Live Status Indicators Bar */}
        <div className="caregiver-header-badges">
          {/* Patient Consent Status Pill */}
          {patientConsent?.isConsented ? (
            <span
              className="caregiver-status-pill pill-active"
              title="Authorized patient & guardian consent on file (§2.7)"
            >
              <span className="pill-dot"></span>
              ✓ Patient Consent on File
            </span>
          ) : (
            <button
              type="button"
              className="caregiver-status-pill pill-consent-required clickable"
              onClick={() => setShowConsentModal(true)}
              title="Click to set up authorized patient consent"
            >
              ⚠️ Patient Consent Required
            </button>
          )}
          <span
            className={`caregiver-status-pill ${
              !isRiskOnline
                ? 'pill-inactive'
                : isRiskConnecting
                ? 'pill-connecting'
                : latestRisk == null && hasHistoricalOnly
                ? 'pill-subtle'
                : latestRisk == null
                ? 'pill-subtle'
                : isTelemetryStale
                ? 'pill-warning'
                : 'pill-active'
            }`}
          >
            <span className="pill-dot"></span>
            {!isRiskOnline
              ? 'Stream Disconnected'
              : isRiskConnecting
              ? 'Connecting Stream...'
              : latestRisk == null && hasHistoricalOnly
              ? 'Socket Open · Standby for Live GPS'
              : latestRisk == null
              ? 'Socket Open · Awaiting First Fix'
              : isTelemetryStale
              ? `Socket Open · GPS Stale (${telemetryTimestamp ? getRelativeTime(telemetryTimestamp, now) : 'delayed'})`
              : 'Telemetry Stream: Synchronized'}
          </span>

          <span className={`caregiver-status-pill ${batteryPillClass}`}>
            {batteryStatusText}
          </span>

          <span className="caregiver-status-pill pill-subtle">
            🛰️ {!isRiskOnline
              ? 'GPS: Stream Offline'
              : !hasValidLocation
              ? 'GPS: Fix Unavailable'
              : gpsAccuracy != null
              ? `GPS ±${gpsAccuracy.toFixed(1)}m${isTelemetryStale ? ' (Stale)' : latestRisk == null && hasHistoricalOnly ? ' (Historical)' : ''}`
              : 'GPS: Unavailable'}
          </span>

          {isAlertIssue && (
            <span className="caregiver-status-pill pill-alert-issue">
              ⚠️ Alerts Sync Issue
            </span>
          )}
        </div>
      </header>

      {/* 2. Serene Hearth Status Hero Banner with Integrated Care Action Strip */}
      <section className={`caregiver-hero-banner ${statusCardClass}`}>
        {/* Top Status & Freshness Row */}
        <div className="caregiver-hero-top-row">
          <div className="hero-status-pill-group">
            <div className={`hero-status-badge ${statusBadgeClass}`}>
              <span className="hero-status-dot-ping"></span>
              <span className="hero-status-title-text">{statusTitle}</span>
            </div>
            <span className="hero-divider-dot">·</span>
            <span className="hero-context-tag">
              {isRiskOffline ? 'Stream Inactive' : `Tier: ${riskTier}`}
            </span>
          </div>

          <div className="hero-telemetry-freshness-group">
            <span className="freshness-chip tabular-nums">
              🛰️ {!isRiskOnline
                ? 'GPS: Offline'
                : !hasValidLocation
                ? 'GPS: Fix Unavailable'
                : gpsAccuracy != null
                ? `GPS ±${gpsAccuracy.toFixed(1)}m HDOP${isTelemetryStale ? ' (Stale)' : latestRisk == null && hasHistoricalOnly ? ' (Historical)' : ''}`
                : 'GPS: Awaiting Fix'}
            </span>
            <span className="freshness-chip tabular-nums">
              🔋 {isBatteryAvailable
                ? `${batteryPct}%${isTelemetryStale ? ' (Stale)' : latestRisk == null && hasHistoricalOnly ? ' (Cached)' : ''}`
                : !isRiskOnline ? 'Battery: Offline' : 'Battery: N/A'}
            </span>
            <span className="freshness-chip age-chip tabular-nums">
              ⏱️ {effectiveRisk?.timestamp ? getRelativeTime(effectiveRisk.timestamp, now) : 'Stream Inactive'}
            </span>
          </div>
        </div>

        {/* Bottom Care Action Card & Quick Utility Actions */}
        <div className="caregiver-hero-action-card">
          <div className="hero-action-content-box">
            <div className="hero-action-tag-row">
              <span className={`hero-action-type-tag ${isActionUrgent ? 'tag-urgent' : 'tag-stable'}`}>
                CARE ACTION: {actionTitle}
              </span>
            </div>
            <p className="hero-action-narrative">
              {actionContent}
            </p>
            {qualitativeNotice && (
              <div className="hero-action-qualitative">
                {qualitativeNotice}
              </div>
            )}
          </div>

          {/* Approved Genuine Application Utilities (Section 6) */}
          <div className="hero-action-buttons-group">
            <button
              type="button"
              className="hero-action-btn hero-btn-primary"
              onClick={() => setMapActionTrigger({ type: 'recenter', key: Date.now() })}
              title="Focus map on latest valid coordinates"
            >
              <span>🎯</span>
              <span>Recenter Map</span>
            </button>
            <button
              type="button"
              className="hero-action-btn hero-btn-secondary"
              onClick={() => setMapActionTrigger({ type: 'focusTrail', key: Date.now() })}
              title="Fit map viewport to full trajectory trail"
            >
              <span>🗺️</span>
              <span>Focus Trail</span>
            </button>
            <button
              type="button"
              className="hero-action-btn hero-btn-outline"
              onClick={onSwitchToTechnical}
              title="Switch to Operator Technical Inspector"
            >
              <span>🛠️</span>
              <span>Technical View</span>
            </button>
          </div>
        </div>
      </section>

      {/* 3. Compact 4-Metric Summary Strip */}
      <div className="caregiver-metrics-strip">
        <div className="caregiver-metric-card">
          <div className="metric-icon-disc metric-disc-shield">🛡️</div>
          <div className="metric-content">
            <span className="metric-eyebrow">Safety Status</span>
            <span className="metric-value">{statusTitle}</span>
            <span className="metric-hint">
              {!isRiskOnline
                ? 'Stream disconnected'
                : isRiskConnecting
                ? 'Connecting stream...'
                : isAwaitingTelemetry
                ? 'Awaiting initial fix'
                : isTelemetryStale
                ? `Risk Tier: ${riskTier} (Stale fix)`
                : latestRisk == null && hasHistoricalOnly
                ? `Risk Tier: ${riskTier} (Historical)`
                : `Risk Tier: ${riskTier}`}
            </span>
          </div>
        </div>

        <div className="caregiver-metric-card">
          <div className="metric-icon-disc metric-disc-pace">⚡</div>
          <div className="metric-content">
            <span className="metric-eyebrow">Movement Pace</span>
            <span className="metric-value">
              {!isRiskOnline
                ? 'Stream Offline'
                : isRiskConnecting
                ? 'Connecting...'
                : isAwaitingTelemetry
                ? 'Awaiting Fix'
                : effectiveRisk?.sensor_metrics?.speed_mps != null
                ? `${(effectiveRisk.sensor_metrics.speed_mps * 3.6).toFixed(1)} km/h`
                : 'Stationary'}
            </span>
            <span className="metric-hint">
              {!isRiskOnline
                ? 'Telemetry offline'
                : isRiskConnecting
                ? 'Connecting to telemetry...'
                : isAwaitingTelemetry
                ? 'Awaiting sensor readings'
                : isTelemetryStale
                ? `Last reported ${telemetryAge || 'previously'}`
                : hasHistoricalOnly
                ? `Last reported ${telemetryAge || 'previously'}`
                : effectiveRisk?.sensor_metrics?.speed_mps != null
                ? `${effectiveRisk.sensor_metrics.speed_mps.toFixed(1)} m/s instantaneous`
                : 'Low activity window'}
            </span>
          </div>
        </div>

        <div className="caregiver-metric-card">
          <div className="metric-icon-disc metric-disc-battery">
            {isLowBattery ? '⚠️' : '🔋'}
          </div>
          <div className="metric-content">
            <span className="metric-eyebrow">Device Power</span>
            <span className="metric-value">
              {!isRiskOnline
                ? 'Stream Offline'
                : isRiskConnecting
                ? 'Connecting...'
                : isBatteryAvailable
                ? `${batteryPct}% Charged`
                : 'Not Reported'}
            </span>
            <span className="metric-hint">
              {!isRiskOnline
                ? 'Stream disconnected'
                : isRiskConnecting
                ? 'Connecting to stream...'
                : isLowBattery
                ? 'Pulse power-saving mode'
                : isTelemetryStale
                ? `Cached from last fix (${telemetryAge || 'stale'})`
                : latestRisk == null && hasHistoricalOnly
                ? 'Cached from historical fix'
                : isLiveTelemetryFresh
                ? 'Live battery telemetry'
                : 'Awaiting telemetry'}
            </span>
          </div>
        </div>

        <div className="caregiver-metric-card">
          <div className="metric-icon-disc metric-disc-trail">📍</div>
          <div className="metric-content">
            <span className="metric-eyebrow">Walking Trail</span>
            <span className="metric-value">
              {locationHistory.length} Waypoint{locationHistory.length === 1 ? '' : 's'}
            </span>
            <span className="metric-hint">
              {locationHistory.length === 0
                ? !isRiskOnline
                  ? 'No recorded trail'
                  : isRiskConnecting
                  ? 'Connecting...'
                  : isCritical
                  ? 'GPS fix unavailable · Risk: CRITICAL'
                  : isSuspicious
                  ? 'GPS fix unavailable · Risk: SUSPICIOUS'
                  : 'Awaiting device fix'
                : !isRiskOnline
                ? `Recorded trail (${locationHistory.length} points)`
                : isTelemetryStale
                ? `Trail paused · ${telemetryAge || 'stale'}`
                : hasHistoricalOnly
                ? `Trail paused · ${telemetryAge || 'historical'}`
                : isLiveTelemetryFresh
                ? '● Live trail tracking'
                : `Recorded trail (${locationHistory.length} points)`}
            </span>
          </div>
        </div>
      </div>

      {/* Main Grid: Map (Left) and Caregiver Insights & Alerts (Right) */}
      <div className="caregiver-grid">
        <div className="caregiver-map-col">
          <MapPanel
            currentLocation={currentLocation}
            locationHistory={locationHistory}
            trailMetadata={trailMetadata}
            currentRiskTier={riskTier}
            currentRiskScore={effectiveRisk?.risk_score}
            userId={userId}
            isCaregiverMode={true}
            isLive={isLiveTelemetryFresh}
            isTelemetryStale={isTelemetryStale}
            hasHistoricalOnly={hasHistoricalOnly}
            isConnecting={isRiskConnecting}
            isOffline={isRiskOffline}
            telemetryTimestamp={telemetryTimestamp}
            telemetryAge={telemetryAge}
            actionTrigger={mapActionTrigger}
            activeCompanion={activeCompanion}
            patientStatus={patientStatus}
          />
        </div>

        <div className="caregiver-side-col">
          {/* Trusted Companion Mode */}
          <TrustedCompanionPanel
            userId={userId}
            currentLocation={currentLocation}
            onCompanionStateChange={setActiveCompanion}
            patientConsent={patientConsent}
            onOpenConsentModal={() => setShowConsentModal(true)}
          />

          {/* Card 1: Plain-Language Movement Observations (Requirement 1) */}
          <div className="panel caregiver-insights-card">
            <div className="panel-header">
              <h2 className="panel-title">Movement Observations</h2>
              <span className="badge-count">
                {isRiskOffline
                  ? 'Stream Offline'
                  : isRiskConnecting
                  ? 'Connecting...'
                  : isAwaitingTelemetry
                  ? 'Awaiting Fix'
                  : isTelemetryStale
                  ? `Stale (${telemetryAge || 'delayed'})`
                  : topFeatures.length > 0
                  ? `${topFeatures.length} factors`
                  : 'Live Insights'}
              </span>
            </div>
            <div className="caregiver-insights-body">
              {topFeatures.length === 0 ? (
                <div className={`caregiver-quiet-observations ${isRiskOffline ? 'quiet-offline' : ''}`}>
                  <div className="quiet-icon-circle">{isRiskOffline ? '⚪' : isRiskConnecting ? '🔄' : isAwaitingTelemetry ? '📡' : '✓'}</div>
                  <div className="quiet-content">
                    <h4 className="quiet-title">
                      {isRiskOffline
                        ? 'Telemetry Stream Offline'
                        : isRiskConnecting
                        ? 'Connecting to Movement Stream'
                        : isAwaitingTelemetry
                        ? 'Awaiting Movement Telemetry'
                        : isTelemetryStale
                        ? 'No Anomalous Dynamics Flagged (Stale)'
                        : 'No Anomalous Dynamics Flagged'}
                    </h4>
                    <p className="quiet-desc">
                      {isRiskOffline
                        ? 'Live movement factors cannot be evaluated while the telemetry stream is disconnected.'
                        : isRiskConnecting
                        ? 'Establishing connection to telemetry stream...'
                        : isAwaitingTelemetry
                        ? 'Connected to stream. Waiting for device sensor fix to evaluate movement dynamics.'
                        : isTelemetryStale
                        ? `Last observation window (${telemetryAge || 'delayed'}) reported no unusual movement. Telemetry is currently delayed.`
                        : 'No unusual acceleration, directional jitter, or pacing patterns flagged in the latest observation window.'}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="caregiver-insights-list">
                  {topFeatures.map((feat, idx) => {
                    const isExpanded = expandedFeatureIndices.has(idx)
                    const isElevated = feat.shap_value != null ? feat.shap_value > 0 : true
                    const featKey = (feat.feature || feat.feature_name || '').toLowerCase()
                    let factorCategory = 'Movement Dynamic'
                    let factorIcon = '📍'
                    if (featKey.includes('speed')) {
                      factorCategory = 'Velocity Analysis'
                      factorIcon = '⚡'
                    } else if (featKey.includes('accel')) {
                      factorCategory = 'Acceleration Pattern'
                      factorIcon = '📈'
                    } else if (featKey.includes('heading') || featKey.includes('turn') || featKey.includes('tortuosity')) {
                      factorCategory = 'Direction & Heading'
                      factorIcon = '🧭'
                    } else if (featKey.includes('displacement') || featKey.includes('anchor')) {
                      factorCategory = 'Location Displacement'
                      factorIcon = '📍'
                    }

                    return (
                      <div
                        key={idx}
                        className={`caregiver-insight-card ${isExpanded ? 'insight-expanded' : ''}`}
                      >
                        <div
                          className="insight-summary-row"
                          role="button"
                          tabIndex={0}
                          aria-expanded={isExpanded}
                          aria-controls={`feature-detail-${idx}`}
                          onClick={() => toggleFeatureExpand(idx)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              toggleFeatureExpand(idx)
                            }
                          }}
                        >
                          <div className="insight-icon-disc">{factorIcon}</div>
                          <div className="insight-title-group">
                            <span className="insight-category-tag">{factorCategory}</span>
                            <span className="insight-text">{feat.human_readable}</span>
                          </div>
                          <span className={`insight-weight-pill ${isElevated ? 'pill-elevated' : 'pill-calm'}`}>
                            {isElevated ? 'Elevated' : 'Steady'}
                          </span>
                          <span className="insight-expand-icon" aria-hidden="true">
                            {isExpanded ? '▲' : '▼'}
                          </span>
                        </div>

                        {isExpanded && (
                          <div id={`feature-detail-${idx}`} className="insight-expanded-body">
                            <div className="insight-detail-row">
                              <span className="insight-detail-label">Factor Focus:</span>
                              <span className="insight-detail-val">{factorCategory}</span>
                            </div>
                            <div className="insight-detail-row">
                              <span className="insight-detail-label">Model Evaluation:</span>
                              <span className={`insight-detail-val ${isElevated ? 'val-elevated' : 'val-calm'}`}>
                                {isElevated ? 'Contributed towards elevated movement alert' : 'Contributed towards stability/calm'}
                              </span>
                            </div>
                            {effectiveRisk?.sensor_metrics?.speed_mps != null && (
                              <div className="insight-detail-row">
                                <span className="insight-detail-label">Current Telemetry Speed:</span>
                                <span className="insight-detail-val">
                                  {effectiveRisk.sensor_metrics.speed_mps.toFixed(1)} m/s ({(effectiveRisk.sensor_metrics.speed_mps * 3.6).toFixed(1)} km/h)
                                </span>
                              </div>
                            )}
                            {effectiveRisk?.timestamp && (
                              <div className="insight-detail-row">
                                <span className="insight-detail-label">Observation Window:</span>
                                <span className="insight-detail-val">
                                  Evaluated at {new Date(effectiveRisk.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}

              <div className="caregiver-status-footnote">
                {isRiskOffline && '• Telemetry stream disconnected. Safety cannot be verified.'}
                {isRiskConnecting && '• Connecting to telemetry stream...'}
                {isAwaitingTelemetry && '• Awaiting initial device fix. Safety cannot be verified.'}
                {isTelemetryStale && `• Telemetry is delayed (${telemetryAge || 'stale'}). Live safety cannot be inferred.`}
                {!isRiskOffline && !isRiskConnecting && !isAwaitingTelemetry && !isTelemetryStale && isQuiescent && '• Stationary or low-activity window.'}
                {!isRiskOffline && !isRiskConnecting && !isAwaitingTelemetry && !isTelemetryStale && isNormalTransit && '• Active movement at a steady pace.'}
                {!isRiskOffline && !isRiskConnecting && !isAwaitingTelemetry && !isTelemetryStale && isSuspicious && '• Pacing, erratic turning, or elevated velocity variance.'}
                {!isRiskOffline && !isRiskConnecting && !isAwaitingTelemetry && !isTelemetryStale && isCritical && '• Sustained high velocity or rapid displacement.'}
              </div>
            </div>
          </div>

          {/* Card 2: Alerts & Safe Route Recalibration (Requirement 2) */}
          <div className="panel caregiver-alerts-card">
            <div className="panel-header">
              <h2 className="panel-title">Active Safety Alerts</h2>
              <span className="badge-count">
                {isAlertOffline
                  ? alertMessages.length === 0
                    ? 'Monitoring Inactive · Connect to Sync Alerts'
                    : `${pendingCount} active · ${dismissedCount} dismissed (Stream Offline)`
                  : isAlertConnecting || alertsLoading
                  ? 'Syncing alerts...'
                  : alertsError && alertMessages.length === 0
                  ? 'Alerts unavailable'
                  : pendingCount === 0
                  ? dismissedCount > 0
                    ? `0 active (${dismissedCount} dismissed)`
                    : '0 active alerts'
                  : `${pendingCount} active alert${pendingCount === 1 ? '' : 's'}${dismissedCount > 0 ? ` (${dismissedCount} dismissed)` : ''}`}
              </span>
            </div>

            {/* Safe Route Recalibration Banner */}
            {recalibrationPrompt && (
              <div className="caregiver-recalibration-card" role="alert">
                <div className="recal-header">
                  <strong>🗺️ Repeated Zone Confirmations · Recalibration Lockout</strong>
                </div>
                <p className="recal-body">
                  {recalibrationPrompt.message ||
                    'Frequent dismissals detected in this zone. Quick dismissal is locked to prevent masking risks. Would you like to update the monitored profile baseline for this zone to adapt to routine travel?'}
                </p>
                <div className="recal-caveat">
                  <em>Note: Updating the profile baseline adjusts alert sensitivity thresholds; it does not verify or guarantee that any route is safe.</em>
                </div>
                <div className="recal-actions">
                  <button
                    type="button"
                    className="recal-btn-primary"
                    onClick={() => handleRecalibrationAnswer('Update Baseline')}
                  >
                    ✓ Update Profile Baseline
                  </button>
                  <button
                    type="button"
                    className="recal-btn-secondary"
                    onClick={() => handleRecalibrationAnswer('Keep Standard')}
                  >
                    Keep Standard Sensitivity
                  </button>
                </div>
              </div>
            )}

            {alertMessages.length === 0 ? (
              <div className={`caregiver-all-clear-card ${emptyAlertCardClass}`}>
                <div className={`all-clear-badge-disc ${emptyAlertDiscClass}`}>{emptyAlertIcon}</div>
                <div className="all-clear-content">
                  <h3 className="all-clear-title">{emptyAlertTitle}</h3>
                  <p className="all-clear-desc">
                    {emptyAlertDesc}
                  </p>
                  <div className="all-clear-meta">
                    <span className="all-clear-pill">{emptyAlertPillText}</span>
                    <span className="all-clear-zone">{emptyAlertZoneText}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="caregiver-alerts-list">
                {alertMessages.map((alert) => {
                  const alertId = alert.alert_id
                  const isDismissing = dismissingIds.has(alertId)
                  const isRecalibrating = recalibrationPrompt?.alertId === alertId
                  const errorMsg = dismissErrors[alertId]
                  const isPending = alert.status === 'PENDING'
                  const isDismissed = alert.status === 'DISMISSED_SAFE'
                  const isExpanded = expandedAlertIds.has(alertId)
                  const dismissalCount =
                    alert.dismissal_count_in_window ??
                    alert.dismissal?.dismissal_count_in_window
                  const sensitivityMultiplier =
                    alert.sensitivity_multiplier ??
                    (dismissalCount != null
                      ? 1.0 - Math.min(0.30, 0.10 * dismissalCount)
                      : null)

                  return (
                    <div
                      key={alertId}
                      className={`caregiver-alert-card ${
                        alert.alert_tier === 3 ? 'alert-tier-3' : 'alert-tier-2'
                      } ${isExpanded ? 'alert-card-expanded' : ''}`}
                    >
                      {/* Top Bar: Severity Badge, Status Pill, Relative Time */}
                      <div className="alert-card-header">
                        <div className="alert-badge-group">
                          <span
                            className={`alert-severity-badge ${
                              alert.alert_tier === 3 ? 'severity-critical' : 'severity-warning'
                            }`}
                          >
                            {alert.alert_tier === 3
                              ? '🚨 Urgent Priority (Tier 3)'
                              : '⚠️ Cautionary (Tier 2)'}
                          </span>
                          <span
                            className={`alert-status-pill ${
                              isDismissed
                                ? 'pill-dismissed'
                                : isPending
                                ? 'pill-pending'
                                : 'pill-other'
                            }`}
                          >
                            {isDismissed
                              ? '✓ Dismissed Safe'
                              : isPending
                              ? '● Action Recommended'
                              : alert.status}
                          </span>
                        </div>
                        <span
                          className="alert-time-stamp"
                          title={alert.timestamp ? new Date(alert.timestamp).toLocaleString() : ''}
                        >
                          {alert.timestamp ? getRelativeTime(alert.timestamp, now) : 'Recently'}
                        </span>
                      </div>

                      {/* Primary Description */}
                      <p className="alert-message-text">
                        {alert.alert_tier === 3
                          ? 'High-risk movement pattern or rapid displacement detected. Prompt check recommended.'
                          : 'Unusual movement pattern, pacing, or elevated velocity variance detected.'}
                      </p>

                      {/* Recommended Action Prompt */}
                      <div className="alert-action-prompt">
                        <span className="prompt-icon">💡</span>
                        <span>
                          Recommended Action: Check on <strong>{personLabel}</strong> directly to confirm their safety.
                        </span>
                      </div>

                      {/* Expand/Collapse Context Toggle */}
                      <div className="alert-expand-bar">
                        <button
                          type="button"
                          className="alert-expand-btn"
                          onClick={() => toggleAlertExpand(alertId)}
                          aria-expanded={isExpanded}
                          aria-controls={`alert-details-${alertId}`}
                        >
                          <span>{isExpanded ? 'Hide Technical Context' : 'View Event Context & Safeguards'}</span>
                          <span className="alert-expand-icon" aria-hidden="true">
                            {isExpanded ? '▲' : '▼'}
                          </span>
                        </button>
                      </div>

                      {/* Expandable Technical Context Panel */}
                      {isExpanded && (
                        <div id={`alert-details-${alertId}`} className="alert-details-panel">
                          <div className="alert-detail-grid">
                            <div className="detail-field">
                              <span className="field-label">Alert Reference:</span>
                              <span className="field-value mono">{alertId}</span>
                            </div>
                            {alert.timestamp && (
                              <div className="detail-field">
                                <span className="field-label">Recorded At:</span>
                                <span className="field-value">
                                  {new Date(alert.timestamp).toLocaleTimeString([], {
                                    hour: '2-digit',
                                    minute: '2-digit',
                                    second: '2-digit',
                                  })}
                                </span>
                              </div>
                            )}
                            {alert.grid_cell_id && (
                              <div className="detail-field">
                                <span className="field-label">Monitored Zone:</span>
                                <span className="field-value mono">{alert.grid_cell_id}</span>
                              </div>
                            )}
                            {sensitivityMultiplier != null && (
                              <div className="detail-field">
                                <span className="field-label">Zone Sensitivity:</span>
                                <span className="field-value">
                                  {(sensitivityMultiplier * 100).toFixed(0)}%
                                  {sensitivityMultiplier < 1
                                    ? ` (${(100 - sensitivityMultiplier * 100).toFixed(0)}% suppression applied)`
                                    : ' (Standard)'}
                                </span>
                              </div>
                            )}
                            {dismissalCount != null && (
                              <div className="detail-field">
                                <span className="field-label">Window Confirmations:</span>
                                <span className="field-value">
                                  {dismissalCount} / 3 before safe-route update
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Error Banner on Failure */}
                      {errorMsg && (
                        <div className="alert-error-banner" role="alert">
                          <span className="error-icon">⚠️</span>
                          <span>{errorMsg}</span>
                        </div>
                      )}

                      {/* Action Footer */}
                      <div className="alert-footer-actions">
                        {isPending ? (
                          <button
                            type="button"
                            className="btn btn-primary alert-confirm-btn"
                            disabled={isDismissing || isRecalibrating}
                            onClick={() => handleDismiss(alert)}
                            aria-busy={isDismissing}
                            aria-label="Confirm status and dismiss alert"
                          >
                            {isDismissing ? (
                              <>
                                <span className="spinner-inline" aria-hidden="true">⏳</span>
                                <span>Confirming & Dismissing...</span>
                              </>
                            ) : isRecalibrating ? (
                              <>
                                <span aria-hidden="true">🔒</span>
                                <span>Baseline Update Required</span>
                              </>
                            ) : (
                              <>
                                <span aria-hidden="true">✓</span>
                                <span>Confirm Status & Dismiss</span>
                              </>
                            )}
                          </button>
                        ) : isDismissed ? (
                          <div className="alert-confirmed-badge">
                            <span aria-hidden="true">✓</span>
                            <span>Dismissed Safe by Caregiver</span>
                          </div>
                        ) : (
                          <span className="status-label">{alert.status}</span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Patient Consent Setup Modal */}
      {showConsentModal && (
        <PatientConsentModal
          userId={userId}
          onClose={() => setShowConsentModal(false)}
          onConsentRecorded={handleConsentRecorded}
        />
      )}
    </div>
  )
}
