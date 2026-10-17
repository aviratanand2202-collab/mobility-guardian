import React, { useState, useEffect, useRef } from 'react'
import CaregiverDashboard from './components/caregiver/CaregiverDashboard'
import TechnicalDashboard from './components/technical/TechnicalDashboard'

export default function App() {
  const [activeView, setActiveView] = useState('caregiver')
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
  const [trailMetadata, setTrailMetadata] = useState([])

  // Historical data (fetched once on connect)
  const [historyEntries, setHistoryEntries] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(null)

  // Alert hydration state (fetched on connect / user switch)
  const [alertsLoading, setAlertsLoading] = useState(false)
  const [alertsError, setAlertsError] = useState(null)

  // Active user ref and abort controller to prevent race conditions on user switch
  const activeUserRef = useRef(userId)
  const alertFetchAbortControllerRef = useRef(null)

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
    if (alertFetchAbortControllerRef.current) {
      alertFetchAbortControllerRef.current.abort()
      alertFetchAbortControllerRef.current = null
    }
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
    setRiskMessages([])
    setAlertMessages([])
    setAlertsLoading(false)
    setAlertsError(null)
    console.info('[Dashboard] disconnectSockets() complete.')
  }

  const connectSockets = (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) return

    const trimmedUser = targetUserId.trim()
    activeUserRef.current = trimmedUser

    disconnectSockets()
    setRiskMessages([])
    setAlertMessages([])
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
          setTrailMetadata((prev) => [
            ...prev,
            {
              timestamp: payload.timestamp || null,
              riskTier: payload.risk_tier || null,
              riskScore: payload.risk_score != null ? payload.risk_score : null,
              speed: payload.sensor_metrics?.speed_mps != null ? payload.sensor_metrics.speed_mps : null,
            },
          ].slice(-30))
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
    fetchAlerts(targetUserId.trim())
  }

  const fetchHistory = async (targetUserId) => {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const res = await fetch(`/api/risk-history/${targetUserId}?limit=100`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }
      const data = await res.json()
      setHistoryEntries(data)

      // Seed location history and trail metadata from historical location data if empty
      const chronological = [...data].sort(
        (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
      )
      const seededCoords = []
      const seededMeta = []
      for (const entry of chronological) {
        if (entry.location && entry.location.lat != null && entry.location.lng != null) {
          seededCoords.push([entry.location.lat, entry.location.lng])
          seededMeta.push({
            timestamp: entry.timestamp || null,
            riskTier: entry.risk_tier || null,
            riskScore: entry.risk_score != null ? entry.risk_score : null,
            speed: entry.sensor_metrics?.speed_mps != null ? entry.sensor_metrics.speed_mps : null,
          })
        }
      }
      if (seededCoords.length > 0) {
        setLocationHistory((prev) => {
          if (prev.length > 0) return prev
          const last = seededCoords[seededCoords.length - 1]
          setCurrentLocation({ lat: last[0], lng: last[1] })
          return seededCoords.slice(-30)
        })
        setTrailMetadata((prev) => {
          if (prev.length > 0) return prev
          return seededMeta.slice(-30)
        })
      }
    } catch (err) {
      console.error('Error fetching risk history:', err)
      setHistoryError(err.message)
    } finally {
      setHistoryLoading(false)
    }
  }

  const fetchAlerts = async (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) return
    const user = targetUserId.trim()

    // Abort previous in-flight alert fetch
    if (alertFetchAbortControllerRef.current) {
      alertFetchAbortControllerRef.current.abort()
    }
    const abortController = new AbortController()
    alertFetchAbortControllerRef.current = abortController

    setAlertsLoading(true)
    setAlertsError(null)

    try {
      const res = await fetch(`/api/alerts/user/${encodeURIComponent(user)}?limit=50`, {
        signal: abortController.signal,
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }
      const data = await res.json()

      // Guard against delayed responses from prior user selections
      if (activeUserRef.current !== user) {
        return
      }

      setAlertMessages((prev) => {
        // Merge persisted alerts with any real-time WebSocket alerts already received
        const alertMap = new Map()
        // 1. Seed with persisted alerts from backend
        for (const alert of data) {
          alertMap.set(alert.alert_id, alert)
        }
        // 2. Overlay any real-time WebSocket alerts already captured in state
        for (const existing of prev) {
          if (alertMap.has(existing.alert_id)) {
            alertMap.set(existing.alert_id, { ...alertMap.get(existing.alert_id), ...existing })
          } else {
            alertMap.set(existing.alert_id, existing)
          }
        }
        // 3. Return sorted descending by timestamp (newest first), max 20
        return Array.from(alertMap.values())
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          .slice(0, 20)
      })
    } catch (err) {
      if (err.name === 'AbortError') {
        // Deliberately aborted due to user switch or newer request
        return
      }
      console.error(`Error fetching alerts for user ${user}:`, err)
      if (activeUserRef.current === user) {
        setAlertsError(err.message)
      }
    } finally {
      if (activeUserRef.current === user) {
        setAlertsLoading(false)
      }
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
  const latestHistory = historyEntries.length > 0 ? historyEntries[historyEntries.length - 1] : null
  const activeRisk = latestRisk || latestHistory

  // Synchronize document.body theme class with active view (Serene Hearth vs Technical)
  useEffect(() => {
    if (activeView === 'caregiver') {
      document.body.classList.remove('caregiver-dark-theme')
      document.body.classList.remove('technical-theme')
      document.body.classList.add('caregiver-serene-theme')
    } else {
      document.body.classList.remove('caregiver-serene-theme')
      document.body.classList.remove('caregiver-dark-theme')
      document.body.classList.add('technical-theme')
    }
    return () => {
      document.body.classList.remove('caregiver-serene-theme')
      document.body.classList.remove('caregiver-dark-theme')
      document.body.classList.remove('technical-theme')
    }
  }, [activeView])

  // Test event listener for state-driven UI verification (gated to dev/test environments)
  useEffect(() => {
    const isTestable =
      Boolean(import.meta.env.DEV) ||
      import.meta.env.MODE === 'test' ||
      (typeof window !== 'undefined' && window.__ENABLE_TEST_HARNESS__ === true)

    if (!isTestable) return

    const handler = (e) => {
      if (e.detail?.riskMessages !== undefined) setRiskMessages(e.detail.riskMessages)
      if (e.detail?.alertMessages !== undefined) setAlertMessages(e.detail.alertMessages)
      if (e.detail?.alertWsStatus !== undefined) setAlertWsStatus(e.detail.alertWsStatus)
      if (e.detail?.riskWsStatus !== undefined) setRiskWsStatus(e.detail.riskWsStatus)
      if (e.detail?.isConnected !== undefined) setIsConnected(e.detail.isConnected)
      if (e.detail?.currentLocation !== undefined) setCurrentLocation(e.detail.currentLocation)
      if (e.detail?.locationHistory !== undefined) setLocationHistory(e.detail.locationHistory)
      if (e.detail?.trailMetadata !== undefined) setTrailMetadata(e.detail.trailMetadata)
    }
    window.addEventListener('__testInjectState', handler)
    return () => window.removeEventListener('__testInjectState', handler)
  }, [])

  return (
    <div className={`container ${activeView === 'caregiver' ? 'caregiver-mode-active' : 'technical-mode-active'}`}>
      {/* Top View Mode Switcher */}
      <div className="view-mode-bar">
        <div className="view-mode-toggle">
          <button
            type="button"
            className={`view-toggle-btn ${activeView === 'caregiver' ? 'active' : ''}`}
            onClick={() => setActiveView('caregiver')}
          >
            🛡️ Caregiver View
          </button>
          <button
            type="button"
            className={`view-toggle-btn ${activeView === 'technical' ? 'active' : ''}`}
            onClick={() => setActiveView('technical')}
          >
            🛠️ Technical Inspector
          </button>
        </div>
      </div>

      {/* Render active interface: pure React render branch without reconnecting */}
      {activeView === 'caregiver' ? (
        <CaregiverDashboard
          userId={userId}
          setUserId={setUserId}
          isConnected={isConnected}
          handleConnectClick={handleConnectClick}
          handleDisconnectClick={handleDisconnectClick}
          handleReconnectClick={handleReconnectClick}
          riskWsStatus={riskWsStatus}
          alertWsStatus={alertWsStatus}
          currentLocation={currentLocation}
          locationHistory={locationHistory}
          trailMetadata={trailMetadata}
          activeRisk={activeRisk}
          latestRisk={latestRisk}
          historyEntries={historyEntries}
          alertMessages={alertMessages}
          handleAlertUpdate={handleAlertUpdate}
          historyLoading={historyLoading}
          historyError={historyError}
          fetchHistory={fetchHistory}
          alertsLoading={alertsLoading}
          alertsError={alertsError}
          fetchAlerts={fetchAlerts}
          onSwitchToTechnical={() => setActiveView('technical')}
        />
      ) : (
        <TechnicalDashboard
          userId={userId}
          setUserId={setUserId}
          isConnected={isConnected}
          handleConnectClick={handleConnectClick}
          handleDisconnectClick={handleDisconnectClick}
          handleReconnectClick={handleReconnectClick}
          riskWsStatus={riskWsStatus}
          alertWsStatus={alertWsStatus}
          currentLocation={currentLocation}
          locationHistory={locationHistory}
          activeRisk={activeRisk}
          latestRisk={latestRisk}
          historyEntries={historyEntries}
          alertMessages={alertMessages}
          handleAlertUpdate={handleAlertUpdate}
          historyLoading={historyLoading}
          historyError={historyError}
          fetchHistory={fetchHistory}
          alertsLoading={alertsLoading}
          alertsError={alertsError}
          fetchAlerts={fetchAlerts}
        />
      )}
    </div>
  )
}
