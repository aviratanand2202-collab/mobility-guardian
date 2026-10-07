import React, { useState, useEffect } from 'react'

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

export default function AlertPanel({ alerts, onAlertUpdate }) {
  const [now, setNow] = useState(Date.now())
  const [dismissingIds, setDismissingIds] = useState(new Set())
  const [dismissErrors, setDismissErrors] = useState({})
  const [recalibrationPrompt, setRecalibrationPrompt] = useState(null)

  // Update relative timestamps every second
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [])

  const handleDismiss = async (alert) => {
    const alertId = alert.alert_id
    if (!alertId || dismissingIds.has(alertId)) return

    // Clear previous error for this alert
    setDismissErrors((prev) => {
      const next = { ...prev }
      delete next[alertId]
      return next
    })

    // Capture previous state snapshot for explicit rollback
    const previousStatus = alert.status
    const previousMultiplier = alert.sensitivity_multiplier

    // Mark as in-flight
    setDismissingIds((prev) => new Set(prev).add(alertId))

    // 1. Optimistic UI update: immediately show DISMISSED_SAFE
    onAlertUpdate(alertId, {
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
        // Rollback optimistic dismiss since alert wasn't dismissed
        onAlertUpdate(alertId, {
          status: previousStatus,
          isOptimistic: false,
        })
        // Trigger recalibration prompt modal/banner
        setRecalibrationPrompt({
          alertId,
          gridCellId: alert.grid_cell_id,
          message: data.message || 'Routine change detected. Add this route to the Safe Mobility Profile permanently?',
        })
        return
      }

      // Case B: Normal dismissal with Stage 4 suppression multiplier
      onAlertUpdate(alertId, {
        status: 'DISMISSED_SAFE',
        sensitivity_multiplier: data.cell_sensitivity_multiplier,
        dismissal_count: data.dismissal_count_in_window,
        isOptimistic: false,
      })
    } catch (err) {
      console.error(`Failed to dismiss alert ${alertId}:`, err)
      // 2. Explicit Rollback on failure: revert to previous status
      onAlertUpdate(alertId, {
        status: previousStatus,
        sensitivity_multiplier: previousMultiplier,
        isOptimistic: false,
      })
      setDismissErrors((prev) => ({
        ...prev,
        [alertId]: `Dismissal failed (${err.message}). State reverted to ${previousStatus}.`,
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
    if (choice === 'Yes') {
      console.info(
        `[Safe Mobility Profile] Caregiver accepted permanent safe-zone recalibration for cell ${recalibrationPrompt?.gridCellId}. Note: Backend endpoint for profile update is pending future stage.`
      )
    }
    setRecalibrationPrompt(null)
  }

  return (
    <div className="panel alert-panel-container">
      <div className="panel-header">
        <h2 className="panel-title">Active Alerts &amp; Quarantine Suppression</h2>
        <span className="badge-count">
          {alerts.length} alert{alerts.length === 1 ? '' : 's'}
        </span>
      </div>

      {/* Recalibration Prompt Banner (4th dismissal flow) */}
      {recalibrationPrompt && (
        <div className="recalibration-banner">
          <div className="recalibration-title">
            ⚠️ Safe Mobility Profile Recalibration
          </div>
          <div className="recalibration-message">
            {recalibrationPrompt.message}
          </div>
          <div className="recalibration-subtext mono">
            Target H3 Cell: {recalibrationPrompt.gridCellId}
          </div>
          <div className="recalibration-actions">
            <button
              className="btn btn-primary"
              style={{ padding: '6px 14px', fontSize: '13px' }}
              onClick={() => handleRecalibrationAnswer('Yes')}
            >
              Yes, Add Permanently
            </button>
            <button
              className="btn btn-secondary"
              style={{ padding: '6px 14px', fontSize: '13px' }}
              onClick={() => handleRecalibrationAnswer('No')}
            >
              No, Dismiss Prompt
            </button>
          </div>
        </div>
      )}

      {alerts.length === 0 ? (
        <div className="empty-state">
          No alerts triggered yet. SUSPICIOUS (Tier 2) or CRITICAL (Tier 3) wandering triggers alerts here.
        </div>
      ) : (
        <div className="alerts-list">
          {alerts.map((alert) => {
            const isDismissing = dismissingIds.has(alert.alert_id)
            const errorMsg = dismissErrors[alert.alert_id]
            const isPending = alert.status === 'PENDING'
            const isDismissed = alert.status === 'DISMISSED_SAFE'

            return (
              <div
                key={alert.alert_id}
                className={`alert-card tier-border-${alert.alert_tier}`}
              >
                <div className="alert-card-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className={`alert-tier-pill tier-${alert.alert_tier === 2 ? 'SUSPICIOUS' : 'CRITICAL'}`}>
                      Tier {alert.alert_tier} ({alert.alert_tier === 2 ? 'SUSPICIOUS' : 'CRITICAL'})
                    </span>
                    <span className="alert-status-text">
                      <strong>{alert.status}</strong>
                    </span>
                  </div>
                  <span className="alert-time-text">
                    {getRelativeTime(alert.timestamp, now)}
                  </span>
                </div>

                <div className="alert-card-body">
                  <div className="alert-info-row">
                    <span className="alert-label">H3 Grid Cell:</span>
                    <span className="mono alert-val">{alert.grid_cell_id}</span>
                  </div>
                  <div className="alert-info-row">
                    <span className="alert-label">Alert ID:</span>
                    <span className="mono alert-val" style={{ fontSize: '11px' }}>
                      {alert.alert_id.slice(0, 12)}...
                    </span>
                  </div>

                  {/* Suppression Multiplier Indicator */}
                  {alert.sensitivity_multiplier != null && (
                    <div className="suppression-pill">
                      🛡️ Suppression Applied: <strong>{alert.sensitivity_multiplier.toFixed(2)}x</strong> multiplier
                      {alert.dismissal_count != null && (
                        <span> (Dismissal #{alert.dismissal_count})</span>
                      )}
                    </div>
                  )}

                  {errorMsg && (
                    <div className="alert-error-banner">
                      {errorMsg}
                    </div>
                  )}
                </div>

                <div className="alert-card-footer">
                  {isPending ? (
                    <button
                      className="btn btn-primary btn-sm"
                      disabled={isDismissing}
                      onClick={() => handleDismiss(alert)}
                    >
                      {isDismissing ? 'Updating Suppression...' : 'Mark User Safe'}
                    </button>
                  ) : isDismissed ? (
                    <span className="dismissed-label">
                      ✓ Caregiver Marked Safe
                    </span>
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
  )
}
