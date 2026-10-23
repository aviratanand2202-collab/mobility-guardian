import React, { useState, useEffect, useRef } from 'react'
import CaregiverDashboard from './components/caregiver/CaregiverDashboard'
import TechnicalDashboard from './components/technical/TechnicalDashboard'
import CompanionInvitePortal from './components/companion/CompanionInvitePortal'
import ModelComparisonDashboard from './components/modelComparison/ModelComparisonDashboard'

export default function App() {
  const [activeView, setActiveView] = useState('caregiver')
  const [userId, setUserId] = useState('test_user_000')
  const [isConnected, setIsConnected] = useState(false)
  const [companionInviteToken, setCompanionInviteToken] = useState(() => {
    if (typeof window !== 'undefined' && window.location?.search) {
      return new URLSearchParams(window.location.search).get('companion_invite') || null
    }
    return null
  })

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
  const [patientStatus, setPatientStatus] = useState(null)
  const [latestWearable, setLatestWearable] = useState(null)
  const [safeZones, setSafeZones] = useState([])
  const [alertsLoading, setAlertsLoading] = useState(false)
  const [alertsError, setAlertsError] = useState(null)

  // Track active fetch ID to discard stale responses from previously selected patients
  const activeFetchIdRef = useRef(0)

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
    setRiskMessages([])
    // Preserve alertMessages so database persisted alerts remain visible across disconnects
    setLatestWearable(null)
    setSafeZones([])
    console.info('[Dashboard] disconnectSockets() complete.')
  }

  const connectSockets = (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) return

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

        // Update wearable biometrics state if present in RiskScoreOutput
        if (payload.wearable_metrics) {
          setLatestWearable(payload.wearable_metrics)
        }

        // Update spatial tracking state if location is present in RiskScoreOutput
        if (payload.location && payload.location.lat != null && payload.location.lng != null) {
          const loc = { lat: payload.location.lat, lng: payload.location.lng }
          setCurrentLocation(loc)
          setLocationHistory((prev) => {
            if (prev.length > 0) {
              const last = prev[prev.length - 1]
              const dLat = Math.abs(last[0] - loc.lat)
              const dLng = Math.abs(last[1] - loc.lng)
              if (dLat > 0.45 || dLng > 0.45) {
                return [[loc.lat, loc.lng]]
              }
            }
            return [...prev, [loc.lat, loc.lng]].slice(-30)
          })
          setTrailMetadata((prev) => {
            if (prev.length > 0) {
              const lastTs = prev[prev.length - 1]?.timestamp
              if (lastTs && payload.timestamp) {
                const diffMin = Math.abs(new Date(payload.timestamp) - new Date(lastTs)) / 60000
                if (diffMin > 30) {
                  return [{
                    timestamp: payload.timestamp || null,
                    riskTier: payload.risk_tier || null,
                    riskScore: payload.risk_score != null ? payload.risk_score : null,
                    speed: payload.sensor_metrics?.speed_mps != null ? payload.sensor_metrics.speed_mps : null,
                  }]
                }
              }
            }
            return [
              ...prev,
              {
                timestamp: payload.timestamp || null,
                riskTier: payload.risk_tier || null,
                riskScore: payload.risk_score != null ? payload.risk_score : null,
                speed: payload.sensor_metrics?.speed_mps != null ? payload.sensor_metrics.speed_mps : null,
              },
            ].slice(-30)
          })
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

  // Synchronize persisted location history whenever selected userId changes
  useEffect(() => {
    if (!userId || !userId.trim()) {
      setCurrentLocation(null)
      setLocationHistory([])
      setTrailMetadata([])
      setHistoryEntries([])
      setPatientStatus(null)
      return
    }
    fetchHistory(userId.trim())
  }, [userId])

  const fetchHistory = async (targetUserId) => {
    if (!targetUserId || !targetUserId.trim()) {
      setHistoryEntries([])
      setCurrentLocation(null)
      setLocationHistory([])
      setTrailMetadata([])
      setPatientStatus(null)
      return
    }

    const trimmedId = targetUserId.trim()
    const fetchId = ++activeFetchIdRef.current

    // On patient change, immediately clear previous patient's displayed location points,
    // markers, waypoints, and risk history before showing the new patient's results
    setCurrentLocation(null)
    setLocationHistory([])
    setTrailMetadata([])
    setHistoryEntries([])
    setRiskMessages([])
    setAlertMessages([])
    setPatientStatus(null)

    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const res = await fetch(`/api/risk-history/${targetUserId}?limit=100`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }
      const data = await res.json()

      // Prevent stale asynchronous responses from previous selection overwriting current one
      if (fetchId !== activeFetchIdRef.current) return

      // Concurrently query patient consent/registry to distinguish registered vs unknown patient
      let exists = false
      let isConsented = false
      try {
        const consentRes = await fetch(`/api/consent/${encodeURIComponent(targetUserId)}`)
        if (consentRes.ok) {
          const consentData = await consentRes.json()
          if (consentData.status !== 'not_found') {
            exists = true
            isConsented = Boolean(consentData.guardian_consent?.granted)
          }
        }
      } catch (err) {
        console.warn('Consent check warning:', err)
      }

      if (fetchId !== activeFetchIdRef.current) return

      if (data.length > 0) {
        exists = true
      }

      setPatientStatus({ exists, isConsented })
      setHistoryEntries(data)

      // Seed location history and trail metadata from historical location data
      const chronological = [...data].sort(
        (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
      )
      const allCoords = []
      const allMeta = []
      for (const entry of chronological) {
        if (entry.location && entry.location.lat != null && entry.location.lng != null) {
          allCoords.push([entry.location.lat, entry.location.lng])
          allMeta.push({
            timestamp: entry.timestamp || null,
            riskTier: entry.risk_tier || null,
            riskScore: entry.risk_score != null ? entry.risk_score : null,
            speed: entry.sensor_metrics?.speed_mps != null ? entry.sensor_metrics.speed_mps : null,
          })
        }
      }

      // Isolate the latest contiguous session segment (prevents connecting distant historical locations like China to India)
      const seededCoords = []
      const seededMeta = []
      if (allCoords.length > 0) {
        const lastIdx = allCoords.length - 1
        let sessionStartIdx = lastIdx
        for (let i = lastIdx - 1; i >= 0; i--) {
          const pt = allCoords[i]
          const nextPt = allCoords[i + 1]
          const dLat = Math.abs(nextPt[0] - pt[0])
          const dLng = Math.abs(nextPt[1] - pt[1])
          const t1 = new Date(allMeta[i]?.timestamp || 0)
          const t2 = new Date(allMeta[i + 1]?.timestamp || 0)
          const timeDiffMin = Math.abs(t2 - t1) / 60000
          if (dLat > 0.45 || dLng > 0.45 || timeDiffMin > 30) {
            sessionStartIdx = i + 1
            break
          }
          sessionStartIdx = i
        }
        for (let i = sessionStartIdx; i <= lastIdx; i++) {
          seededCoords.push(allCoords[i])
          seededMeta.push(allMeta[i])
        }
      }
      if (seededCoords.length > 0) {
        const last = seededCoords[seededCoords.length - 1]
        setCurrentLocation({ lat: last[0], lng: last[1] })
        setLocationHistory(seededCoords.slice(-30))
        setTrailMetadata(seededMeta.slice(-30))
      } else {
        setCurrentLocation(null)
        setLocationHistory([])
        setTrailMetadata([])
      }

      // Query latest wearable biometrics for patient
      try {
        const wearRes = await fetch(`/api/wearable/${trimmedId}/latest`)
        if (wearRes.ok) {
          const wearData = await wearRes.json()
          if (fetchId === activeFetchIdRef.current) {
            if (wearData.status === 'SUCCESS' && wearData.reading) {
              setLatestWearable({
                ...wearData.reading,
                evidence: wearData.evidence,
                age_seconds: wearData.age_seconds,
                is_stale: wearData.is_stale,
              })
            } else {
              setLatestWearable(null)
            }
          }
        }
      } catch (wearErr) {
        console.warn('Wearable fetch non-blocking warning:', wearErr)
      }

      // Query patient safe zones from database
      try {
        const szRes = await fetch(`/api/safe-zones/${encodeURIComponent(trimmedId)}`)
        if (szRes.ok) {
          const szData = await szRes.json()
          if (fetchId === activeFetchIdRef.current) {
            setSafeZones(Array.isArray(szData) ? szData : [])
          }
        } else {
          if (fetchId === activeFetchIdRef.current) {
            setSafeZones([])
          }
        }
      } catch (szErr) {
        console.warn('Safe zones fetch non-blocking warning:', szErr)
        if (fetchId === activeFetchIdRef.current) {
          setSafeZones([])
        }
      }

      // Query recent alerts for patient from database
      try {
        const alRes = await fetch(`/api/alerts/user/${encodeURIComponent(trimmedId)}?limit=20`)
        if (alRes.ok) {
          const alData = await alRes.json()
          if (fetchId === activeFetchIdRef.current && Array.isArray(alData)) {
            setAlertMessages(alData)
          }
        }
      } catch (alErr) {
        console.warn('Alerts fetch non-blocking warning:', alErr)
      }
    } catch (err) {
      if (fetchId !== activeFetchIdRef.current) return
      console.error('Error fetching risk history:', err)
      setHistoryError(err.message)
      setCurrentLocation(null)
      setLocationHistory([])
      setTrailMetadata([])
      setHistoryEntries([])
      setPatientStatus({ exists: false, isConsented: false })
      setLatestWearable(null)
      setSafeZones([])
    } finally {
      if (fetchId === activeFetchIdRef.current) {
        setHistoryLoading(false)
      }
    }
  }


  // Polling helper to keep active alerts synchronized with backend SQLite database
  const fetchAlerts = async (targetId) => {
    const id = (targetId || userId || '').trim()
    if (!id) return
    setAlertsLoading(true)
    try {
      const res = await fetch(`/api/alerts/user/${encodeURIComponent(id)}?limit=20`)
      if (res.ok) {
        const data = await res.json()
        if (Array.isArray(data)) {
          setAlertMessages(data)
          setAlertsError(null)
        }
      } else {
        setAlertsError(`HTTP ${res.status}: Failed to load active alerts`)
      }
    } catch (err) {
      console.warn('Alert polling warning:', err)
      setAlertsError(err.message)
    } finally {
      setAlertsLoading(false)
    }
  }

  // Periodic polling for active alerts and companion state
  useEffect(() => {
    if (!userId || !userId.trim()) return
    fetchAlerts(userId)
    const interval = setInterval(() => {
      fetchAlerts(userId)
    }, 2500)
    return () => clearInterval(interval)
  }, [userId])

  const refreshSafeZones = async (targetId) => {
    const id = targetId || userId
    if (!id || !id.trim()) return
    try {
      const res = await fetch(`/api/safe-zones/${encodeURIComponent(id.trim())}`)
      if (res.ok) {
        const data = await res.json()
        setSafeZones(Array.isArray(data) ? data : [])
      }
    } catch (err) {
      console.warn('Refresh safe zones warning:', err)
    }
  }

  const handleAlertUpdate = (alertId, updateFields) => {
    setAlertMessages((prev) =>
      prev.map((a) => (a.alert_id === alertId ? { ...a, ...updateFields } : a))
    )
  }

  const handleConnectClick = (e, explicitId) => {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault()
      e.stopPropagation()
    }
    const targetId = (typeof explicitId === 'string' && explicitId.trim())
      ? explicitId.trim()
      : (typeof e === 'string' && e.trim())
      ? e.trim()
      : (userId || '').trim()

    if (!targetId) {
      console.warn('[Dashboard] Connect aborted: patient ID is empty.')
      return
    }

    console.info('[Dashboard] handleConnectClick() fired for patient:', targetId)
    if (targetId !== userId) {
      setUserId(targetId)
    }

    if (isConnected) {
      if (targetId === userId) {
        console.warn('[Dashboard] handleConnectClick() suppressed: already connected.')
        return
      }
      disconnectSockets()
    }

    connectSockets(targetId)
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
      {/* Companion Invitation Portal (Active when invite token is in URL) */}
      {companionInviteToken && (
        <CompanionInvitePortal
          inviteToken={companionInviteToken}
          onClose={() => {
            setCompanionInviteToken(null)
            if (typeof window !== 'undefined' && window.history?.replaceState) {
              window.history.replaceState({}, '', window.location.pathname)
            }
          }}
        />
      )}

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
          <button
            type="button"
            className={`view-toggle-btn ${activeView === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveView('comparison')}
          >
            📊 Model Comparison
          </button>
        </div>
      </div>

      {/* Render active interface: pure React render branch without reconnecting */}
      {activeView === 'comparison' ? (
        <ModelComparisonDashboard />
      ) : activeView === 'caregiver' ? (
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
          patientStatus={patientStatus}
          latestWearable={latestWearable}
          safeZones={safeZones}
          refreshSafeZones={refreshSafeZones}
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
          trailMetadata={trailMetadata}
          activeRisk={activeRisk}
          latestRisk={latestRisk}
          historyEntries={historyEntries}
          alertMessages={alertMessages}
          handleAlertUpdate={handleAlertUpdate}
          historyLoading={historyLoading}
          historyError={historyError}
          fetchHistory={fetchHistory}
          patientStatus={patientStatus}
          latestWearable={latestWearable}
          safeZones={safeZones}
          refreshSafeZones={refreshSafeZones}
          alertsLoading={alertsLoading}
          alertsError={alertsError}
          fetchAlerts={fetchAlerts}
        />
      )}
    </div>
  )
}
