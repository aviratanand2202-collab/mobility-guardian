import React, { useState, useEffect, useRef } from 'react'
import MapPanel from './components/MapPanel'
import RiskPanel from './components/RiskPanel'
import AlertPanel from './components/AlertPanel'

export default function App() {
  const [userId, setUserId] = useState('sim_stage6')
  const [isConnected, setIsConnected] = useState(false)

  // Socket status states: 'disconnected' | 'connecting' | 'open' | 'closed' | 'error'
  const [riskWsStatus, setRiskWsStatus] = useState('disconnected')
  const [alertWsStatus, setAlertWsStatus] = useState('disconnected')

  // Stream message arrays (newest first, max 20)
  const [riskMessages, setRiskMessages] = useState([])
  const [alertMessages, setAlertMessages] = useState([])

  // Spatial location state (decoupled from riskMessages feed)
  const [currentLocation, setCurrentLocation] = useState(null)
  const [locationHistory, setLocationHistory] = useState([])

  // Historical data (fetched once on connect)
  const [historyEntries, setHistoryEntries] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(null)

  // Refs to hold WebSocket instances
  const riskSocketRef = useRef(null)
  const alertSocketRef = useRef(null)

  // Clean up sockets on unmount
  useEffect(() => {
    return () => {
      disconnectSockets()
    }
  }, [])

  const disconnectSockets = () => {
    console.info('[Dashboard] disconnectSockets() executing: closing sockets and setting isConnected=false')
    if (riskSocketRef.current) {
      riskSocketRef.current.onclose = null
      riskSocketRef.current.close()
      riskSocketRef.current = null
    }
    if (alertSocketRef.current) {
      alertSocketRef.current.onclose = null
      alertSocketRef.current.close()
      alertSocketRef.current = null
    }
    setRiskWsStatus('disconnected')
    setAlertWsStatus('disconnected')
    setIsConnected(false)
    console.info('[Dashboard] disconnectSockets() complete.')
  }

  const connectSockets = (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) return

    disconnectSockets()
    setIsConnected(true)
    setRiskWsStatus('connecting')
    setAlertWsStatus('connecting')

    // Connect via Vite dev proxy host (window.location.host, typically localhost:5173)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host || 'localhost:5173'

    const riskUrl = `${protocol}//${host}/ws/risk/${targetUserId.trim()}`
    const alertsUrl = `${protocol}//${host}/ws/alerts/${targetUserId.trim()}`

    // 1. Risk WebSocket
    const riskWs = new WebSocket(riskUrl)
    riskSocketRef.current = riskWs

    riskWs.onopen = () => {
      setRiskWsStatus('open')
    }
    riskWs.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        setRiskMessages((prev) => [payload, ...prev].slice(0, 20))

        // Update spatial tracking state if location is present in RiskScoreOutput
        if (payload.location && payload.location.lat != null && payload.location.lng != null) {
          const loc = { lat: payload.location.lat, lng: payload.location.lng }
          setCurrentLocation(loc)
          setLocationHistory((prev) => [...prev, [loc.lat, loc.lng]].slice(-30))
        }
      } catch (err) {
        console.error('Failed to parse risk websocket message', err)
      }
    }
    riskWs.onerror = () => {
      setRiskWsStatus('error')
    }
    riskWs.onclose = () => {
      setRiskWsStatus('closed')
    }

    // 2. Alerts WebSocket
    const alertWs = new WebSocket(alertsUrl)
    alertSocketRef.current = alertWs

    alertWs.onopen = () => {
      setAlertWsStatus('open')
    }
    alertWs.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        setAlertMessages((prev) => {
          // Check if alert already exists in list (e.g. repeated pending in same cell)
          const existingIdx = prev.findIndex((a) => a.alert_id === payload.alert_id)
          if (existingIdx !== -1) {
            const updated = [...prev]
            updated[existingIdx] = { ...updated[existingIdx], ...payload }
            return updated
          }
          return [payload, ...prev].slice(0, 20)
        })
      } catch (err) {
        console.error('Failed to parse alert websocket message', err)
      }
    }
    alertWs.onerror = () => {
      setAlertWsStatus('error')
    }
    alertWs.onclose = () => {
      setAlertWsStatus('closed')
    }

    // 3. Fetch recent history via REST (through Vite proxy /api)
    fetchHistory(targetUserId.trim())
  }

  const fetchHistory = async (targetUserId) => {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const res = await fetch(`/api/risk-history/${targetUserId}?limit=20`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }
      const data = await res.json()
      setHistoryEntries(data)

      // Seed location history from historical location data if empty
      setLocationHistory((prev) => {
        if (prev.length > 0) return prev
        const seeded = []
        // Ensure chronological order (oldest first -> newest last)
        const chronological = [...data].sort(
          (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
        )
        for (const entry of chronological) {
          if (entry.location && entry.location.lat != null && entry.location.lng != null) {
            seeded.push([entry.location.lat, entry.location.lng])
          }
        }
        if (seeded.length > 0) {
          const last = seeded[seeded.length - 1]
          setCurrentLocation({ lat: last[0], lng: last[1] })
        }
        return seeded.slice(-30)
      })
    } catch (err) {
      console.error('Error fetching risk history:', err)
      setHistoryError(err.message)
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleAlertUpdate = (alertId, updateFields) => {
    setAlertMessages((prev) =>
      prev.map((a) => (a.alert_id === alertId ? { ...a, ...updateFields } : a))
    )
  }

  const handleConnectClick = (e) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    console.info('[Dashboard] handleConnectClick() fired. Current isConnected state:', isConnected)
    if (isConnected) {
      console.warn('[Dashboard] handleConnectClick() suppressed: already connected (prevented form resubmission).')
      return
    }
    connectSockets(userId)
  }

  const handleDisconnectClick = (e) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    console.info('[Dashboard] handleDisconnectClick() fired: calling disconnectSockets()')
    disconnectSockets()
  }

  const handleReconnectClick = (e) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    connectSockets(userId)
  }

  const latestRisk = riskMessages.length > 0 ? riskMessages[0] : null

  return (
    <div className="container">
      <header className="header">
        <h1>Predictive Geofencing Operator Dashboard</h1>
        <p>Stage 8: Live Spatial Visualization, TreeSHAP Explainability &amp; Quarantine Suppression</p>
      </header>

      {/* Connection & Controls Bar */}
      <div className="controls-card">
        <form onSubmit={handleConnectClick} className="form-group">
          <label htmlFor="user-id-input">User ID:</label>
          <input
            id="user-id-input"
            type="text"
            className="form-input"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="e.g. sim_stage6"
            disabled={isConnected}
          />
          {!isConnected ? (
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

      {/* Main Dashboard Grid: Map on Left (60%), Risk & Alerts on Right (40%) */}
      <div className="dashboard-main-grid">
        <div className="map-column">
          <MapPanel
            currentLocation={currentLocation}
            locationHistory={locationHistory}
            currentRiskTier={latestRisk?.risk_tier}
            currentRiskScore={latestRisk?.risk_score}
            userId={userId}
          />
        </div>

        <div className="side-column">
          <RiskPanel
            currentRisk={latestRisk}
            historyEntries={historyEntries}
          />

          <AlertPanel
            alerts={alertMessages}
            onAlertUpdate={handleAlertUpdate}
          />
        </div>
      </div>

      {/* Bottom Panel: Historical Risk Records (REST) */}
      <div className="panel" style={{ marginTop: '24px' }}>
        <div className="panel-header">
          <h2 className="panel-title">
            Historical Risk Records (REST: GET /api/risk-history/{userId}?limit=20)
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
            {isConnected
              ? 'No history found for this user in the database.'
              : 'Connect to query database history for this user.'}
          </div>
        )}

        {!historyLoading && !historyError && historyEntries.length > 0 && (
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
                </tr>
              </thead>
              <tbody>
                {historyEntries.map((entry) => (
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
