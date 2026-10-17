import React, { useState, useEffect, useRef, useMemo } from 'react'
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  CircleMarker,
  Tooltip,
  Circle,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'

// Color map for risk tiers per spec
const TIER_COLORS = {
  QUIESCENT: '#3b82f6', // Vivid Blue
  NORMAL_TRANSIT: '#10b981', // Emerald Teal
  SUSPICIOUS: '#f59e0b', // Warm Amber
  CRITICAL: '#ef4444', // High-Risk Red
}

const TIER_LABELS = {
  QUIESCENT: 'Stationary / Low Activity',
  NORMAL_TRANSIT: 'Normal Steady Pace',
  SUSPICIOUS: 'Unusual Movement Pattern',
  CRITICAL: 'High Risk / Attention Recommended',
}

// Curated palette for real multi-patient trajectory comparison
const MULTI_PATIENT_PALETTE = [
  { color: '#2563eb', dashArray: null, badge: 'P1' },        // Royal Blue (solid)
  { color: '#059669', dashArray: '6, 6', badge: 'P2' },       // Emerald (dashed)
  { color: '#d97706', dashArray: '3, 6', badge: 'P3' },       // Amber (dotted)
  { color: '#7c3aed', dashArray: '8, 4, 2, 4', badge: 'P4' }, // Purple (dash-dot)
  { color: '#db2777', dashArray: '12, 6', badge: 'P5' },      // Rose (long dash)
  { color: '#0891b2', dashArray: '4, 4', badge: 'P6' },       // Cyan (short dash)
]

// Pure Haversine distance in meters
function haversineMeters(coord1, coord2) {
  if (!coord1 || !coord2) return 0
  const [lat1, lon1] = coord1
  const [lat2, lon2] = coord2
  const R = 6371000
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return Math.round(R * c * 10) / 10
}

// Helpers for truthful presentation
function formatTimestamp(isoString) {
  if (!isoString) return 'Timestamp unavailable'
  try {
    const date = new Date(isoString)
    if (isNaN(date.getTime())) return 'Timestamp unavailable'
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return 'Timestamp unavailable'
  }
}

function formatRelativeTime(isoString, now) {
  if (!isoString) return ''
  try {
    const d = new Date(isoString).getTime()
    if (isNaN(d)) return ''
    const diffSec = Math.floor((now - d) / 1000)
    if (diffSec < 5) return 'just now'
    if (diffSec < 60) return `${diffSec}s ago`
    const diffMin = Math.floor(diffSec / 60)
    if (diffMin < 60) return `${diffMin}m ago`
    const diffHr = Math.floor(diffMin / 60)
    return `${diffHr}h ago`
  } catch {
    return ''
  }
}

function formatSpeed(speedMps) {
  if (speedMps == null || isNaN(speedMps)) return 'Speed not recorded'
  const mps = Number(speedMps)
  const kmh = mps * 3.6
  return `${mps.toFixed(1)} m/s (${kmh.toFixed(1)} km/h)`
}

// Custom DivIcon creator for monitored patient
function createRiskMarkerIcon(riskTier, isCaregiverMode = false) {
  let color = TIER_COLORS[riskTier] || '#6366f1'
  if (isCaregiverMode) {
    if (riskTier === 'QUIESCENT') color = '#3b82f6'
    else if (riskTier === 'NORMAL_TRANSIT') color = '#244a38'
    else if (riskTier === 'SUSPICIOUS') color = '#d97736'
    else if (riskTier === 'CRITICAL') color = '#ba1a1a'
  }
  const html = `
    <div class="custom-live-marker-wrapper">
      <div class="custom-live-marker-halo" style="background-color: ${color};"></div>
      <div class="custom-live-marker-pin" style="background-color: ${color};">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="7" r="4" fill="#ffffff" />
          <path d="M5.5 21v-2a6.5 6.5 0 0 1 13 0v2" />
        </svg>
      </div>
      <div class="custom-live-marker-tip" style="border-top-color: ${color};"></div>
    </div>
  `
  return L.divIcon({
    className: 'custom-risk-marker',
    html,
    iconSize: [36, 46],
    iconAnchor: [18, 44],
    popupAnchor: [0, -44],
  })
}

// Custom DivIcon for real Multi-Patient Tracking (clearly tied to real Patient ID)
function createMultiPatientMarkerIcon(patientId, color, badge, pointCount) {
  const html = `
    <div class="custom-live-marker-wrapper multi-patient-marker-wrapper">
      <div class="custom-live-marker-halo" style="background-color: ${color}; color: ${color};"></div>
      <div class="custom-live-marker-pin" style="background-color: ${color}; color: #ffffff; font-weight: 800; font-size: 11px;">
        ${badge}
      </div>
      <div class="custom-live-marker-tip" style="border-top-color: ${color};"></div>
      <div class="multi-patient-pill-tag" style="background-color: ${color};">${patientId}</div>
    </div>
  `
  return L.divIcon({
    className: 'custom-risk-marker custom-multi-patient-marker',
    html,
    iconSize: [44, 56],
    iconAnchor: [22, 54],
    popupAnchor: [0, -44],
  })
}

// Custom DivIcon for Trusted Companion / Family Member
function createCompanionMarkerIcon(isStale = false, relationship = 'trusted_companion') {
  const isFamily = relationship === 'family'
  const color = isStale ? '#94a3b8' : (isFamily ? '#2563eb' : '#8b5cf6')
  const icon = isFamily ? '👤' : '🤝'
  const tag = isStale ? 'STALE' : (isFamily ? 'FAMILY' : 'COMPANION')
  const html = `
    <div class="custom-live-marker-wrapper companion-marker-wrapper">
      <div class="custom-live-marker-halo" style="background-color: ${color}; color: ${color};"></div>
      <div class="custom-live-marker-pin" style="background-color: ${color}; color: #ffffff; font-weight: 800; font-size: 14px;">
        ${icon}
      </div>
      <div class="custom-live-marker-tip" style="border-top-color: ${color};"></div>
      <div class="companion-pill-tag" style="background-color: ${color};">${tag}</div>
    </div>
  `
  return L.divIcon({
    className: 'custom-companion-marker',
    html,
    iconSize: [38, 48],
    iconAnchor: [19, 46],
    popupAnchor: [0, -44],
  })
}

// Controller component to smoothly fly/pan/fit map when viewTarget changes
function MapViewController({ viewTarget }) {
  const map = useMap()
  useEffect(() => {
    if (!viewTarget) return
    if (viewTarget.mode === 'fitBounds' && viewTarget.bounds) {
      map.fitBounds(viewTarget.bounds, { padding: [40, 40], maxZoom: 17 })
    } else if (viewTarget.coords && viewTarget.coords[0] != null && viewTarget.coords[1] != null) {
      if (viewTarget.mode === 'fly') {
        map.flyTo(viewTarget.coords, 16, { animate: true, duration: 0.8 })
      } else {
        map.panTo(viewTarget.coords, { animate: true, duration: 0.4 })
      }
    }
  }, [map, viewTarget])
  return null
}

// Controller component for following live position when not inspecting historical point
function MapLiveFollowController({ activePosition, isInspecting }) {
  const map = useMap()
  useEffect(() => {
    if (activePosition && activePosition[0] != null && activePosition[1] != null && !isInspecting) {
      map.panTo(activePosition, { animate: true, duration: 0.5 })
    }
  }, [map, activePosition, isInspecting])
  return null
}

// Controller component to fit bounds when in Multi-Patient view
function MapMultiPatientBoundsController({ isMultiPatientView, bounds }) {
  const map = useMap()
  useEffect(() => {
    if (isMultiPatientView && bounds) {
      map.fitBounds(bounds, { padding: [45, 45], maxZoom: 16 })
    }
  }, [map, isMultiPatientView, bounds])
  return null
}

export default function MapPanel({
  currentLocation,
  locationHistory = [],
  trailMetadata = [],
  currentRiskTier,
  currentRiskScore,
  userId,
  isCaregiverMode = false,
  isLive = true,
  isTelemetryStale = false,
  hasHistoricalOnly = false,
  isConnecting = false,
  isOffline = false,
  telemetryTimestamp = null,
  telemetryAge = '',
  actionTrigger = null,
  activeCompanion = null,
  configuredSafeZone = null,
  patientStatus = null,
}) {
  const [selectedPointIndex, setSelectedPointIndex] = useState(null)
  const [viewTarget, setViewTarget] = useState(null)
  const [now, setNow] = useState(Date.now())
  const timelineRef = useRef(null)

  // Real Multi-Patient Tracking view state (replaces fake Group Demo)
  const [isMultiPatientView, setIsMultiPatientView] = useState(false)
  const [multiPatientIds, setMultiPatientIds] = useState(['test_user_000', 'test_user_2'])
  const [newPatientInput, setNewPatientInput] = useState('')
  const [multiPatientData, setMultiPatientData] = useState({})
  const [multiPatientLoading, setMultiPatientLoading] = useState(false)

  // When patient selection changes, sync multiPatientIds to include the active userId
  useEffect(() => {
    if (userId && userId.trim()) {
      const cleanId = userId.trim()
      setMultiPatientIds((prev) => (prev.includes(cleanId) ? prev : [cleanId, ...prev]))
    }
  }, [userId])

  // Fetch real records independently for each patient in multiPatientIds
  useEffect(() => {
    if (!isMultiPatientView) return

    let isMounted = true
    const fetchAllMultiPatients = async () => {
      setMultiPatientLoading(true)
      const results = {}

      for (let i = 0; i < multiPatientIds.length; i++) {
        const pid = multiPatientIds[i].trim()
        if (!pid) continue
        const palette = MULTI_PATIENT_PALETTE[i % MULTI_PATIENT_PALETTE.length]

        try {
          const res = await fetch(`/api/risk-history/${encodeURIComponent(pid)}?limit=100`)
          let data = []
          if (res.ok) {
            data = await res.json()
          }

          let exists = data.length > 0
          try {
            const cRes = await fetch(`/api/consent/${encodeURIComponent(pid)}`)
            if (cRes.ok) {
              const cData = await cRes.json()
              if (cData.status !== 'not_found') {
                exists = true
              }
            }
          } catch {
            // ignore consent query error
          }

          const validCoords = []
          const validMeta = []
          const chronological = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))

          for (const entry of chronological) {
            if (entry.location && entry.location.lat != null && entry.location.lng != null) {
              validCoords.push([entry.location.lat, entry.location.lng])
              validMeta.push({
                timestamp: entry.timestamp,
                riskTier: entry.risk_tier,
                riskScore: entry.risk_score,
                speed: entry.sensor_metrics?.speed_mps,
              })
            }
          }

          results[pid] = {
            id: pid,
            color: palette.color,
            dashArray: palette.dashArray,
            badge: palette.badge,
            records: data,
            coords: validCoords,
            trailMetadata: validMeta,
            exists,
            hasData: validCoords.length > 0,
          }
        } catch (err) {
          results[pid] = {
            id: pid,
            color: palette.color,
            dashArray: palette.dashArray,
            badge: palette.badge,
            records: [],
            coords: [],
            trailMetadata: [],
            exists: false,
            hasData: false,
            error: err.message,
          }
        }
      }

      if (isMounted) {
        setMultiPatientData(results)
        setMultiPatientLoading(false)
      }
    }

    fetchAllMultiPatients()
    return () => {
      isMounted = false
    }
  }, [isMultiPatientView, multiPatientIds])

  // Derive all valid coordinates across all loaded patients in multi-patient mode
  const allMultiPatientCoords = useMemo(() => {
    if (!isMultiPatientView) return []
    const coords = []
    Object.values(multiPatientData).forEach((p) => {
      if (p.coords && p.coords.length > 0) {
        coords.push(...p.coords)
      }
    })
    return coords
  }, [isMultiPatientView, multiPatientData])

  const multiPatientBounds = useMemo(() => {
    if (allMultiPatientCoords.length > 0) {
      return L.latLngBounds(allMultiPatientCoords)
    }
    return null
  }, [allMultiPatientCoords])

  // Active single-patient position
  const activePosition =
    currentLocation && currentLocation.lat != null && currentLocation.lng != null
      ? [currentLocation.lat, currentLocation.lng]
      : locationHistory.length > 0
      ? locationHistory[locationHistory.length - 1]
      : null

  // Explicit toggle handler to switch between Single Patient and Multi-Patient Tracking
  const handleToggleMultiPatientView = (enabled) => {
    setIsMultiPatientView(enabled)
    setSelectedPointIndex(null)
    if (enabled) {
      if (multiPatientBounds) {
        setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })
      }
    } else {
      if (locationHistory && locationHistory.length > 1) {
        const bounds = L.latLngBounds(locationHistory)
        setViewTarget({ bounds, mode: 'fitBounds', key: Date.now() })
      } else if (activePosition) {
        setViewTarget({ coords: activePosition, mode: 'fly', key: Date.now() })
      }
    }
  }

  // Multi-patient addition and removal handlers
  const handleAddMultiPatient = (e) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    const trimmed = (newPatientInput || '').trim()
    if (!trimmed) return
    if (!multiPatientIds.includes(trimmed)) {
      setMultiPatientIds((prev) => [...prev, trimmed])
    }
    setNewPatientInput('')
  }

  const handleRemoveMultiPatient = (idToRemove) => {
    setMultiPatientIds((prev) => prev.filter((id) => id !== idToRemove))
  }

  // When patient selection or locationHistory changes, automatically focus the real patient's trajectory
  useEffect(() => {
    if (isMultiPatientView) return
    setSelectedPointIndex(null)
    if (locationHistory && locationHistory.length > 1) {
      const bounds = L.latLngBounds(locationHistory)
      setViewTarget({ bounds, mode: 'fitBounds', key: Date.now() })
    } else if (activePosition) {
      setViewTarget({ coords: activePosition, mode: 'fly', key: Date.now() })
    }
  }, [userId, activePosition?.[0], activePosition?.[1], isMultiPatientView, locationHistory.length])

  // Refresh relative time every 5 seconds
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 5000)
    return () => clearInterval(timer)
  }, [])

  // Auto-scroll timeline to selected card when selectedPointIndex changes
  useEffect(() => {
    if (selectedPointIndex !== null && timelineRef.current) {
      const selectedEl = timelineRef.current.querySelector(
        `[data-waypoint-index="${selectedPointIndex}"]`
      )
      if (selectedEl) {
        selectedEl.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        })
      }
    }
  }, [selectedPointIndex])

  // Center coordinate determination (purely derived from actual coordinates)
  const center = isMultiPatientView
    ? (allMultiPatientCoords.length > 0 ? allMultiPatientCoords[0] : null)
    : (activePosition || (locationHistory.length > 0 ? locationHistory[0] : null))

  const riskTier = currentRiskTier || 'QUIESCENT'
  const markerIcon = createRiskMarkerIcon(riskTier, isCaregiverMode)
  const trailColor = TIER_COLORS[riskTier] || '#2563eb'

  const personLabel = userId && userId.trim() ? `User (${userId.trim()})` : 'Monitored Person'
  const caregiverStatusText =
    riskTier === 'QUIESCENT'
      ? 'All Looks Well'
      : riskTier === 'NORMAL_TRANSIT'
      ? 'Normal Movement'
      : riskTier === 'SUSPICIOUS'
      ? 'Unusual Movement Pattern'
      : 'Attention Recommended'

  // Recenter handler
  const handleRecenter = () => {
    if (isMultiPatientView) {
      if (multiPatientBounds) {
        setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })
      }
      return
    }
    if (activePosition) {
      setSelectedPointIndex(null)
      setViewTarget({ coords: activePosition, mode: 'fly', key: Date.now() })
    }
  }

  // Focus Trail handler
  const handleFocusTrail = () => {
    if (isMultiPatientView) {
      if (multiPatientBounds) {
        setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })
      }
      return
    }
    if (locationHistory && locationHistory.length > 1) {
      setSelectedPointIndex(null)
      const bounds = L.latLngBounds(locationHistory)
      setViewTarget({ bounds, mode: 'fitBounds', key: Date.now() })
    } else if (activePosition) {
      handleRecenter()
    }
  }

  // External action trigger listener
  useEffect(() => {
    if (!actionTrigger) return
    if (actionTrigger.type === 'recenter') {
      handleRecenter()
    } else if (actionTrigger.type === 'focusTrail') {
      handleFocusTrail()
    }
  }, [actionTrigger])

  // Select trail point handler
  const handleSelectPoint = (idx) => {
    setSelectedPointIndex(idx)
    const targetCoords = locationHistory[idx]
    if (targetCoords) {
      setViewTarget({ coords: targetCoords, mode: 'pan', key: Date.now() })
    }
  }

  // Dismiss inspector handler
  const handleCloseInspector = () => {
    setSelectedPointIndex(null)
    if (activePosition) {
      setViewTarget({ coords: activePosition, mode: 'fly', key: Date.now() })
    }
  }

  // Keyboard shortcut: Escape closes inspector
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && selectedPointIndex !== null) {
        handleCloseInspector()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedPointIndex, activePosition])

  const selectedCoords =
    selectedPointIndex !== null && locationHistory[selectedPointIndex]
      ? locationHistory[selectedPointIndex]
      : null
  const selectedMeta =
    selectedPointIndex !== null && trailMetadata[selectedPointIndex]
      ? trailMetadata[selectedPointIndex]
      : {}

  return (
    <div className="panel map-panel-container">
      {/* Panel Header */}
      <div className="panel-header">
        <div className="map-header-left">
          <h2 className="panel-title">
            {isMultiPatientView
              ? 'Multi-Patient Tracking · Trajectory Comparison'
              : isCaregiverMode
              ? !activePosition
                ? riskTier === 'CRITICAL'
                  ? 'Spatial Tracking (GPS Unavailable · Critical Risk)'
                  : riskTier === 'SUSPICIOUS'
                  ? 'Spatial Tracking (GPS Unavailable · Elevated Risk)'
                  : 'Awaiting Initial Telemetry'
                : isLive
                ? 'Live Location & Walking Trail'
                : isTelemetryStale
                ? 'Last-Known Location & Trail'
                : hasHistoricalOnly
                ? 'Cached Location & Trail'
                : 'Last-Known Location & Trail'
              : 'Spatial Tracking & Trajectory Map'}
          </h2>
          {isMultiPatientView ? (
            <span className="map-header-sector-badge" style={{ color: '#059669', borderColor: '#10b981' }}>
              Historical Trajectories · {allMultiPatientCoords.length} pts across {multiPatientIds.filter((id) => multiPatientData[id]?.hasData).length} users
            </span>
          ) : isCaregiverMode ? (
            <span className="map-header-sector-badge">
              {personLabel} · {!activePosition
                ? riskTier === 'CRITICAL'
                  ? '🚨 Critical Risk (No Fix)'
                : riskTier === 'SUSPICIOUS'
                ? '⚠️ Unusual Movement (No Fix)'
                : isConnecting
                ? 'Connecting...'
                : isOffline
                ? 'Offline State'
                : 'Awaiting Satellite Fix'
              : isLive
                ? 'Active Live GPS'
                : isTelemetryStale
                ? `Cached Fix (${telemetryAge || 'Stale'})`
                : hasHistoricalOnly
                ? 'Cached Fix (Historical)'
                : isConnecting
                ? 'Connecting...'
                : 'Offline State'}
            </span>
          ) : (
            activePosition ? (
              <span className="mono" style={{ fontSize: '11px', color: '#6b7280' }}>
                ({activePosition[0].toFixed(5)}, {activePosition[1].toFixed(5)})
              </span>
            ) : (
              <span className="map-header-sector-badge" style={{ color: '#94a3b8', borderColor: '#475569' }}>
                No recorded location data for this patient{patientStatus && !patientStatus.exists ? ' (Unknown ID)' : ''}
              </span>
            )
          )}
        </div>

        <div className="map-header-toolbar">
          {/* Explicit Mode Toggle: Single Patient vs Real Multi-Patient View */}
          <div className="map-mode-toggle-group" role="group" aria-label="Monitoring Mode">
            <button
              type="button"
              className={`map-mode-toggle-btn ${!isMultiPatientView ? 'active' : ''}`}
              onClick={() => handleToggleMultiPatientView(false)}
              aria-pressed={!isMultiPatientView}
            >
              Single Patient
            </button>
            <button
              type="button"
              className={`map-mode-toggle-btn ${isMultiPatientView ? 'active' : ''}`}
              onClick={() => handleToggleMultiPatientView(true)}
              aria-pressed={isMultiPatientView}
            >
              👥 Multi-Patient View
            </button>
          </div>

          {isMultiPatientView ? (
            <button
              type="button"
              className="map-header-btn map-header-fit-group-btn"
              onClick={() => multiPatientBounds && setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })}
              disabled={!multiPatientBounds}
              title="Fit map bounds so all compared patient trajectories are visible"
              aria-label="Fit all patient trajectories"
            >
              <span>🗺️</span>
              <span>Fit All Patients</span>
            </button>
          ) : isCaregiverMode ? (
            <>
              <button
                type="button"
                className="map-header-btn map-header-recenter-btn"
                onClick={handleRecenter}
                disabled={!activePosition}
                title={
                  !activePosition
                    ? 'No coordinates available'
                    : isLive
                    ? 'Recenter map on active live position'
                    : 'Recenter map on last known position'
                }
                aria-label={isLive ? 'Recenter on live person' : 'Recenter on last known location'}
              >
                <span>🎯</span>
                <span>Recenter</span>
              </button>
              <button
                type="button"
                className="map-header-btn map-header-focus-btn"
                onClick={handleFocusTrail}
                disabled={!activePosition && (!locationHistory || locationHistory.length === 0)}
                title={
                  !activePosition && (!locationHistory || locationHistory.length === 0)
                    ? 'No coordinates available'
                    : 'Fit map to complete walking trail'
                }
                aria-label="Focus trail bounds"
              >
                <span>🗺️</span>
                <span>Focus Trail</span>
              </button>
            </>
          ) : null}

          <span className="badge-count tabular-nums">
            {isMultiPatientView
              ? `${multiPatientIds.length} patients (${allMultiPatientCoords.length} pts)`
              : isCaregiverMode
              ? isLive && activePosition
                ? `${locationHistory.length} trail point${locationHistory.length === 1 ? '' : 's'}`
                : `${locationHistory.length} point${locationHistory.length === 1 ? '' : 's'}${isTelemetryStale ? ' (stale)' : hasHistoricalOnly ? ' (cached)' : ' (historical)'}`
              : `${locationHistory.length} trail point${locationHistory.length === 1 ? '' : 's'}`}
          </span>
        </div>
      </div>

      {/* Real Multi-Patient Trajectory Comparison Dashboard Strip */}
      {isMultiPatientView && (
        <div className="multi-patient-dashboard-strip" role="region" aria-label="Multi-Patient Comparison Controls">
          <div className="multi-patient-controls-row">
            <form onSubmit={handleAddMultiPatient} className="multi-patient-add-form">
              <label htmlFor="multi-patient-input" className="sr-only">Add Patient ID:</label>
              <input
                id="multi-patient-input"
                type="text"
                className="multi-patient-input"
                placeholder="Add patient ID to compare (e.g. test_user_2)..."
                value={newPatientInput}
                onChange={(e) => setNewPatientInput(e.target.value)}
              />
              <button type="submit" className="multi-patient-add-btn">
                + Add Patient
              </button>
            </form>
            <div className="multi-patient-actions">
              <button
                type="button"
                className="map-header-btn map-header-fit-group-btn"
                onClick={() => multiPatientBounds && setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })}
                disabled={!multiPatientBounds}
                title="Fit map to all displayed patient trajectories"
              >
                <span>🗺️</span>
                <span>Fit All Patients</span>
              </button>
            </div>
          </div>

          {/* Active Patients Chips */}
          <div className="multi-patient-chips-row">
            {multiPatientIds.map((pid, idx) => {
              const p = multiPatientData[pid]
              const color = p?.color || MULTI_PATIENT_PALETTE[idx % MULTI_PATIENT_PALETTE.length].color
              return (
                <div key={pid} className="multi-patient-chip" style={{ borderColor: color }}>
                  <span className="chip-dot" style={{ backgroundColor: color }} />
                  <span className="chip-label">{pid}</span>
                  <span className="chip-status">
                    {multiPatientLoading && !p
                      ? 'Loading...'
                      : p?.hasData
                      ? `${p.coords.length} pts`
                      : p?.exists
                      ? '0 pts (No data)'
                      : 'Unknown ID'}
                  </span>
                  <button
                    type="button"
                    className="chip-remove-btn"
                    onClick={() => handleRemoveMultiPatient(pid)}
                    aria-label={`Remove patient ${pid}`}
                    title={`Remove ${pid} from comparison`}
                  >
                    ✕
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Map Canvas Wrapper */}
      <div className="map-wrapper">
        {isMultiPatientView && allMultiPatientCoords.length === 0 ? (
          <div className="map-empty-telemetry-container" role="status" aria-label="No recorded location data for selected patients">
            <div className="map-radar-graphic">
              <div className="radar-circle radar-circle-outer"></div>
              <div className="radar-circle radar-circle-mid"></div>
              <div className="radar-circle radar-circle-inner"></div>
              <div className="radar-scanner"></div>
              <div className="radar-center-beacon">
                <span className="beacon-icon">👥</span>
              </div>
            </div>
            <div className="map-empty-telemetry-content">
              <span className="map-empty-pill">Multi-Patient Tracking</span>
              <h3 className="map-empty-title">No recorded location data for selected patients</h3>
              <p className="map-empty-desc">
                None of the selected patients ({multiPatientIds.join(', ') || 'none selected'}) have recorded location points in the database.
                Enter patient IDs with saved history (e.g. test_user_000, test_user_2) to view and compare trajectories.
              </p>
              <div className="map-empty-tags">
                {multiPatientIds.map((pid) => (
                  <span key={pid} className="empty-tag">
                    {pid}: <strong>{multiPatientData[pid]?.exists ? '0 points' : 'Unknown ID'}</strong>
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : !isMultiPatientView && isCaregiverMode && !activePosition ? (
          <div className={`map-empty-telemetry-container ${
            riskTier === 'CRITICAL' && isCaregiverMode
              ? 'map-empty-critical'
              : riskTier === 'SUSPICIOUS' && isCaregiverMode
              ? 'map-empty-suspicious'
              : ''
          }`}>
            <div className="map-radar-graphic">
              <div className="radar-circle radar-circle-outer"></div>
              <div className="radar-circle radar-circle-mid"></div>
              <div className="radar-circle radar-circle-inner"></div>
              <div className="radar-scanner"></div>
              <div className="radar-center-beacon">
                <span className="beacon-icon">
                  {riskTier === 'CRITICAL' && isCaregiverMode ? '🚨' : riskTier === 'SUSPICIOUS' && isCaregiverMode ? '⚠️' : '📍'}
                </span>
              </div>
            </div>
            <div className="map-empty-telemetry-content">
              {riskTier === 'CRITICAL' && isCaregiverMode ? (
                <>
                  <span className="map-empty-pill">🚨 Critical Movement Risk · Location Unavailable</span>
                  <h3 className="map-empty-title">Satellite Coordinates Unavailable</h3>
                  <p className="map-empty-desc">
                    Movement telemetry reports high-risk motion (Tier 3), but satellite GPS coordinates are currently unavailable. Do not wait for a map fix — check on {personLabel} directly.
                  </p>
                  <div className="map-empty-tags">
                    <span className="empty-tag">User: <strong>{personLabel}</strong></span>
                    <span className="empty-tag">Risk: <strong style={{ color: '#ba1a1a' }}>CRITICAL (Tier 3)</strong></span>
                    <span className="empty-tag">GPS: <strong>Fix Unavailable</strong></span>
                  </div>
                </>
              ) : riskTier === 'SUSPICIOUS' && isCaregiverMode ? (
                <>
                  <span className="map-empty-pill">⚠️ Unusual Movement Flagged · Location Unavailable</span>
                  <h3 className="map-empty-title">Satellite Coordinates Unavailable</h3>
                  <p className="map-empty-desc">
                    Movement telemetry reports unusual motion patterns (Tier 2), but satellite GPS coordinates are currently unavailable. Continue monitoring closely and verify {personLabel}&apos;s safety.
                  </p>
                  <div className="map-empty-tags">
                    <span className="empty-tag">User: <strong>{personLabel}</strong></span>
                    <span className="empty-tag">Risk: <strong style={{ color: '#b45309' }}>SUSPICIOUS (Tier 2)</strong></span>
                    <span className="empty-tag">GPS: <strong>Fix Unavailable</strong></span>
                  </div>
                </>
              ) : (
                <>
                  <span className="map-empty-pill">
                    Patient: {userId || 'Unspecified'}
                    {patientStatus && !patientStatus.exists ? ' · Unknown ID' : ''}
                  </span>
                  <h3 className="map-empty-title">No recorded location data for this patient</h3>
                  <p className="map-empty-desc">
                    {patientStatus && !patientStatus.exists
                      ? `Unknown patient ID (${userId || 'unspecified'}). No registered patient record or consent is on file in the database.`
                      : `No recorded location data for this patient (${userId || 'unspecified'}). Replay or stream patient telemetry to view location history and map trajectory.`}
                  </p>
                  <div className="map-empty-tags">
                    <span className="empty-tag">Patient ID: <strong>{userId || 'None'}</strong></span>
                    <span className="empty-tag">Status: <strong>{patientStatus && !patientStatus.exists ? 'Unknown Patient ID (Unregistered)' : 'No recorded location data for this patient'}</strong></span>
                    <span className="empty-tag">Trail: <strong>0 points</strong></span>
                  </div>
                </>
              )}
            </div>
          </div>
        ) : !isMultiPatientView && !activePosition ? (
          <div className="map-empty-telemetry-container" role="status" aria-label="No recorded location data">
            <div className="map-radar-graphic">
              <div className="radar-circle radar-circle-outer"></div>
              <div className="radar-circle radar-circle-mid"></div>
              <div className="radar-circle radar-circle-inner"></div>
              <div className="radar-scanner"></div>
              <div className="radar-center-beacon">
                <span className="beacon-icon">📍</span>
              </div>
            </div>
            <div className="map-empty-telemetry-content">
              <span className="map-empty-pill">
                Patient: {userId || 'Unspecified'}
                {patientStatus && !patientStatus.exists ? ' · Unknown ID' : ''}
              </span>
              <h3 className="map-empty-title">No recorded location data for this patient</h3>
              <p className="map-empty-desc">
                {patientStatus && !patientStatus.exists
                  ? `Unknown patient ID (${userId || 'unspecified'}). No registered patient record or consent is on file in the database.`
                  : `No recorded location data for this patient (${userId || 'unspecified'}). Replay or stream patient telemetry to view location history and map trajectory.`}
              </p>
              <div className="map-empty-tags">
                <span className="empty-tag">Patient ID: <strong>{userId || 'None'}</strong></span>
                <span className="empty-tag">Status: <strong>{patientStatus && !patientStatus.exists ? 'Unknown Patient ID (Unregistered)' : 'No recorded location data for this patient'}</strong></span>
                <span className="empty-tag">Trail: <strong>0 points</strong></span>
              </div>
            </div>
          </div>
        ) : (
          <MapContainer
            center={center}
            zoom={16}
            scrollWheelZoom={true}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <MapViewController viewTarget={viewTarget} />
            <MapLiveFollowController
              activePosition={isMultiPatientView ? null : activePosition}
              isInspecting={selectedPointIndex !== null}
            />
            <MapMultiPatientBoundsController
              isMultiPatientView={isMultiPatientView}
              bounds={multiPatientBounds}
            />

            {/* When Multi-Patient View active: render real trajectory for each added patient */}
            {isMultiPatientView ? (
              <>
                {Object.values(multiPatientData).map((p) => {
                  if (!p.coords || p.coords.length === 0) return null
                  const latestCoord = p.coords[p.coords.length - 1]
                  const pIcon = createMultiPatientMarkerIcon(p.id, p.color, p.badge, p.coords.length)

                  return (
                    <React.Fragment key={p.id}>
                      {/* Distinct Polyline Trail */}
                      {p.coords.length > 1 && (
                        <Polyline
                          positions={p.coords}
                          pathOptions={{
                            color: p.color,
                            weight: 4,
                            opacity: 0.85,
                            dashArray: p.dashArray || undefined,
                            lineJoin: 'round',
                          }}
                        />
                      )}

                      {/* Waypoint CircleMarkers */}
                      {p.coords.map((pt, pIdx) => {
                        const isEnd = pIdx === p.coords.length - 1
                        const meta = p.trailMetadata[pIdx] || {}
                        return (
                          <CircleMarker
                            key={`${p.id}-pt-${pIdx}`}
                            center={pt}
                            radius={isEnd ? 7 : 4.5}
                            pathOptions={{
                              fillColor: p.color,
                              fillOpacity: isEnd ? 1 : 0.7,
                              color: '#ffffff',
                              weight: isEnd ? 2 : 1,
                            }}
                          >
                            <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                              <div style={{ fontSize: '11px', fontWeight: 600 }}>
                                Patient: {p.id} · Point #{pIdx + 1}
                                {isEnd ? ' (Latest Recorded)' : ''}
                                <br />
                                {formatTimestamp(meta.timestamp)}
                                <br />
                                <span style={{ fontSize: '10px', color: '#64748b' }}>
                                  {pt[0].toFixed(5)}, {pt[1].toFixed(5)}
                                </span>
                              </div>
                            </Tooltip>
                          </CircleMarker>
                        )
                      })}

                      {/* Prominent Marker with real Patient ID */}
                      <Marker position={latestCoord} icon={pIcon}>
                        <Tooltip direction="top" offset={[0, -44]} opacity={0.95}>
                          <div style={{ fontWeight: 700, fontSize: '12px' }}>
                            👤 Patient: {p.id}
                            <div style={{ fontSize: '10px', color: '#64748b' }}>
                              {p.coords.length} recorded points
                            </div>
                          </div>
                        </Tooltip>
                        <Popup>
                          <div style={{ fontSize: '12px', lineHeight: 1.5, minWidth: '190px' }}>
                            <strong style={{ color: p.color, fontSize: '13px' }}>Patient: {p.id}</strong>
                            <div style={{ marginTop: '4px' }}>
                              <strong>Recorded Points:</strong> {p.coords.length}
                            </div>
                            <div>
                              <strong>Latest Lat/Lng:</strong> {latestCoord[0].toFixed(5)}, {latestCoord[1].toFixed(5)}
                            </div>
                            {p.trailMetadata.length > 0 && p.trailMetadata[p.trailMetadata.length - 1]?.timestamp && (
                              <div>
                                <strong>Latest Recorded:</strong> {formatTimestamp(p.trailMetadata[p.trailMetadata.length - 1].timestamp)}
                              </div>
                            )}
                            <div style={{ marginTop: '6px', fontSize: '10px', color: '#64748b', fontStyle: 'italic' }}>
                              Historical recorded trajectory · Not live GPS
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    </React.Fragment>
                  )
                })}
              </>
            ) : (
              <>
                {/* Configured Patient Safe Zone Boundary (only when configured for this patient) */}
                {configuredSafeZone && configuredSafeZone.configured && configuredSafeZone.center && (
                  <Circle
                    center={configuredSafeZone.center}
                    radius={configuredSafeZone.radius_m || 200}
                    pathOptions={{
                      color: '#2563eb',
                      fillColor: '#3b82f6',
                      fillOpacity: 0.08,
                      dashArray: '6, 8',
                      weight: 2,
                    }}
                  >
                    <Tooltip direction="top" opacity={0.95}>
                      <div style={{ fontSize: '11px', fontWeight: 600 }}>
                        🛡️ {configuredSafeZone.name || 'Configured Patient Safe Zone'}
                        <br />
                        <span style={{ fontSize: '10px', color: '#64748b' }}>
                          Radius: {configuredSafeZone.radius_m}m · Configured Safe Boundary
                        </span>
                      </div>
                    </Tooltip>
                  </Circle>
                )}

                {/* Trail Polyline */}
                {locationHistory.length > 1 && (
                  <Polyline
                    positions={locationHistory}
                    pathOptions={{
                      color: isCaregiverMode ? '#244a38' : trailColor,
                      weight: isCaregiverMode ? 3.5 : 4,
                      opacity: 0.85,
                      dashArray: isCaregiverMode ? '4, 6' : '3, 6',
                      lineJoin: 'round',
                    }}
                  />
                )}

                {/* Selectable Trail CircleMarkers */}
                {locationHistory.map((coords, idx) => {
                  const meta = trailMetadata[idx] || {}
                  const isSelected = selectedPointIndex === idx
                  const isLatest = idx === locationHistory.length - 1
                  let dotColor = '#52634f'
                  if (meta.riskTier === 'QUIESCENT') dotColor = '#3b82f6'
                  else if (meta.riskTier === 'NORMAL_TRANSIT') dotColor = '#244a38'
                  else if (meta.riskTier === 'SUSPICIOUS') dotColor = '#d97736'
                  else if (meta.riskTier === 'CRITICAL') dotColor = '#ba1a1a'

                  return (
                    <CircleMarker
                      key={`trail-marker-${idx}`}
                      center={coords}
                      radius={isSelected ? 10 : 6}
                      pathOptions={{
                        fillColor: dotColor,
                        fillOpacity: isSelected ? 1 : 0.85,
                        color: isSelected ? '#ffffff' : '#244a38',
                        weight: isSelected ? 3 : 1.5,
                      }}
                      eventHandlers={{
                        click: () => handleSelectPoint(idx),
                      }}
                    >
                      <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                        <div style={{ fontSize: '11px', fontWeight: 600 }}>
                          Point #{idx + 1} {isLatest ? '(Latest)' : ''}
                          <br />
                          {formatTimestamp(meta.timestamp)}
                        </div>
                      </Tooltip>
                    </CircleMarker>
                  )
                })}

                {/* Active / Last Known Position Prominent Marker */}
                {activePosition && (
                  <Marker position={activePosition} icon={markerIcon}>
                    <Popup>
                      {isCaregiverMode ? (
                        <div style={{ fontSize: '13px', lineHeight: 1.5 }}>
                          <strong>{personLabel}</strong>
                          <br />
                          <span>{isLive ? 'Active Live Position' : 'Last Known Position'}</span>
                          <br />
                          <span>
                            Status: {isLive ? caregiverStatusText : 'Monitoring Interrupted (Offline)'}
                          </span>
                        </div>
                      ) : (
                        <div style={{ fontSize: '12px', lineHeight: 1.4 }}>
                          <strong>User:</strong> {userId}
                          <br />
                          <strong>Risk Tier:</strong> {riskTier}
                          <br />
                          <strong>Score:</strong>{' '}
                          {currentRiskScore != null ? Number(currentRiskScore).toFixed(1) : '-'}
                          <br />
                          <strong>Lat/Lng:</strong> {activePosition[0].toFixed(5)},{' '}
                          {activePosition[1].toFixed(5)}
                        </div>
                      )}
                    </Popup>
                  </Marker>
                )}

                {/* Trusted Companion / Associated Family Member Marker & Connecting Line */}
                {activeCompanion && activeCompanion.status === 'ACTIVE' && activeCompanion.last_companion_lat != null && activeCompanion.last_companion_lng != null && (
                  <>
                    {activePosition && (
                      <Polyline
                        positions={[activePosition, [activeCompanion.last_companion_lat, activeCompanion.last_companion_lng]]}
                        pathOptions={{
                          color: activeCompanion.is_location_stale
                            ? '#94a3b8'
                            : activeCompanion.accompaniment_status === 'Separated'
                            ? '#f59e0b'
                            : activeCompanion.relationship === 'family'
                            ? '#2563eb'
                            : '#8b5cf6',
                          weight: 2.5,
                          dashArray: activeCompanion.accompaniment_status === 'Separated' ? '6, 6' : '5, 8',
                          opacity: 0.85,
                        }}
                      />
                    )}
                    <Marker
                      position={[activeCompanion.last_companion_lat, activeCompanion.last_companion_lng]}
                      icon={createCompanionMarkerIcon(activeCompanion.is_location_stale, activeCompanion.relationship)}
                    >
                      <Tooltip direction="top" offset={[0, -44]} opacity={0.95}>
                        <div style={{ fontWeight: 700, fontSize: '12px' }}>
                          {activeCompanion.relationship === 'family' ? '👤 ' : '🤝 '}
                          {activeCompanion.companion_alias || (activeCompanion.companion_user_id ? `Family (${activeCompanion.companion_user_id})` : 'Trusted Companion')}
                          <div style={{ fontSize: '10px', color: '#64748b' }}>
                            {activeCompanion.relationship === 'family' ? 'Associated Family Member' : 'Temporary Companion · Personal ID Protected'}
                          </div>
                        </div>
                      </Tooltip>
                      <Popup>
                        <div style={{ fontSize: '12px', lineHeight: 1.5, minWidth: '200px' }}>
                          <strong style={{ color: activeCompanion.relationship === 'family' ? '#2563eb' : '#8b5cf6', fontSize: '13px' }}>
                            {activeCompanion.relationship === 'family' ? '👤 ' : '🤝 '}
                            {activeCompanion.companion_alias || (activeCompanion.companion_user_id ? `Family Member (${activeCompanion.companion_user_id})` : 'Trusted Companion')}
                          </strong>
                          <div style={{ fontSize: '11px', color: '#64748b' }}>
                            {activeCompanion.relationship === 'family' ? 'Authorized Family Member' : 'Temporary Accompaniment Session'}
                          </div>
                          <div style={{ marginTop: '6px' }}>
                            <strong>Accompaniment:</strong>{' '}
                            <span style={{
                              fontWeight: 700,
                              color: activeCompanion.accompaniment_status === 'With Companion'
                                ? '#059669'
                                : activeCompanion.accompaniment_status === 'Separated'
                                ? '#d97706'
                                : '#64748b',
                            }}>
                              {activeCompanion.accompaniment_status || 'Accompaniment Unverified'}
                            </span>
                          </div>
                          {activeCompanion.separation_distance_m != null && (
                            <div>
                              <strong>Distance to Patient:</strong> {Math.round(activeCompanion.separation_distance_m)}m (Threshold: 50m)
                            </div>
                          )}
                          <div>
                            <strong>Telemetry:</strong>{' '}
                            {activeCompanion.is_location_stale ? '⚠️ Stale Telemetry (>60s)' : '🟢 Fresh Live Fix'}
                          </div>
                          {activeCompanion.last_companion_timestamp && (
                            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                              Timestamp: {formatTimestamp(activeCompanion.last_companion_timestamp)}
                            </div>
                          )}
                          {activeCompanion.safe_zone_departure_accompanied && (
                            <div style={{ fontSize: '10.5px', color: '#2563eb', marginTop: '4px', background: '#eff6ff', padding: '3px 6px', borderRadius: '4px' }}>
                              🛡️ Accompanied Safe Zone Departure (Together with patient — no family separation)
                            </div>
                          )}
                        </div>
                      </Popup>
                    </Marker>
                  </>
                )}
              </>
            )}
          </MapContainer>
        )}

        {/* Truthful Corner Calibration Badge */}
        {isMultiPatientView ? (
          <div className="map-corner-badge">
            <div className="map-corner-badge-item">
              <span className="map-corner-dot dot-multi"></span>
              <span className="map-corner-text" style={{ fontWeight: 700 }}>
                Multi-Patient Trajectory Comparison (Historical Data)
              </span>
              <span className="map-corner-divider">|</span>
              <span className="map-corner-text">
                {multiPatientIds.length} Patients Configured ({allMultiPatientCoords.length} pts)
              </span>
            </div>
          </div>
        ) : isCaregiverMode && activePosition && (
          <div className="map-corner-badge">
            <div className="map-corner-badge-item">
              <span className={`map-corner-dot ${isLive ? 'dot-live' : isTelemetryStale ? 'dot-stale' : 'dot-offline'}`}></span>
              <span className="map-corner-text">
                {isLive
                  ? 'Live GPS Stream'
                  : isTelemetryStale
                  ? `Last Known Fix (${telemetryAge || 'Stale'})`
                  : hasHistoricalOnly
                  ? 'Cached Location'
                  : 'Last Known Fix'}
              </span>
            </div>
            {locationHistory.length > 0 && (
              <span className="map-corner-trail-info">
                <span className="map-corner-divider">|</span>
                <span className="map-corner-badge-item">
                  <span className="map-corner-trail-dash"></span>
                  <span className="map-corner-text">
                    {isLive
                      ? `${locationHistory.length} Trail Points`
                      : isTelemetryStale
                      ? `Trail paused · ${telemetryAge || 'stale'}`
                      : `${locationHistory.length} Trail Points (paused)`}
                  </span>
                </span>
              </span>
            )}
          </div>
        )}

        {/* Floating Controls Overlay */}
        {isMultiPatientView ? (
          multiPatientBounds && (
            <div className="map-floating-controls">
              <button
                type="button"
                className="map-floating-recenter-btn map-floating-focus-btn"
                onClick={() => setViewTarget({ bounds: multiPatientBounds, mode: 'fitBounds', key: Date.now() })}
                aria-label="Fit all patient trajectories"
                title="Fit map bounds to all compared patient trajectories"
              >
                <span className="recenter-icon">🗺️</span>
                <span className="recenter-text">
                  <span className="recenter-text-full">Fit All Patients</span>
                  <span className="recenter-text-mobile">Fit All</span>
                </span>
              </button>
            </div>
          )
        ) : isCaregiverMode && activePosition && (
          <div className="map-floating-controls">
            <button
              type="button"
              className="map-floating-recenter-btn"
              onClick={handleRecenter}
              disabled={!activePosition}
              aria-label={
                isLive
                  ? 'Recenter map on live person location'
                  : 'Recenter map on last known person location'
              }
              title={
                !activePosition
                  ? 'No coordinates available'
                  : isLive
                  ? 'Recenter map on active live position'
                  : 'Recenter map on last known position'
              }
            >
              <span className="recenter-icon">🎯</span>
              <span className="recenter-text">
                <span className="recenter-text-full">Recenter ({isLive ? 'Live' : 'Last Known'})</span>
                <span className="recenter-text-mobile">Recenter</span>
              </span>
            </button>
            {locationHistory.length > 1 && (
              <button
                type="button"
                className="map-floating-recenter-btn map-floating-focus-btn"
                onClick={handleFocusTrail}
                aria-label="Focus full walking trail"
                title="Fit map viewport to full trajectory trail"
              >
                <span className="recenter-icon">🗺️</span>
                <span className="recenter-text">
                  <span className="recenter-text-full">Focus Trail</span>
                  <span className="recenter-text-mobile">Trail</span>
                </span>
              </button>
            )}
          </div>
        )}

        {/* Contextual Point-Inspector Card */}
        {!isMultiPatientView && isCaregiverMode && selectedCoords && (
          <div
            className="map-point-inspector"
            role="region"
            aria-label="Selected trail point details"
          >
            <div className="inspector-drag-handle" aria-hidden="true" />
            <div className="inspector-header">
              <div className="inspector-title-wrap">
                <div className="inspector-icon-disc">
                  <span>📍</span>
                </div>
                <div className="inspector-title-col">
                  <div className="inspector-title-row">
                    <span className="inspector-title-text">
                      Waypoint #{selectedPointIndex + 1}
                    </span>
                    {selectedPointIndex === locationHistory.length - 1 && (
                      <span className="inspector-latest-tag">
                        {isLive ? '● Live Current' : '● Last Known'}
                      </span>
                    )}
                  </div>
                  <span className="inspector-subtext">
                    {formatTimestamp(selectedMeta.timestamp)}
                    {formatRelativeTime(selectedMeta.timestamp, now) && (
                      <span> · {formatRelativeTime(selectedMeta.timestamp, now)}</span>
                    )}
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="inspector-close-btn"
                onClick={handleCloseInspector}
                aria-label="Close point inspector"
                title="Close inspector"
              >
                ✕
              </button>
            </div>

            <div className="inspector-metrics-grid">
              <div className="inspector-metric-col">
                <span className="inspector-metric-label">Status</span>
                <span
                  className="inspector-tier-pill"
                  style={{
                    backgroundColor: `${TIER_COLORS[selectedMeta.riskTier] || '#64748b'}25`,
                    color: TIER_COLORS[selectedMeta.riskTier] || '#334155',
                    borderColor: TIER_COLORS[selectedMeta.riskTier] || '#94a3b8',
                  }}
                >
                  {TIER_LABELS[selectedMeta.riskTier] || 'Unclassified'}
                </span>
              </div>

              <div className="inspector-metric-col inspector-metric-divider">
                <span className="inspector-metric-label">Speed</span>
                <span className="inspector-metric-val tabular-nums">
                  {formatSpeed(selectedMeta.speed)}
                </span>
              </div>

              <div className="inspector-metric-col inspector-metric-divider">
                <span className="inspector-metric-label">Coordinates</span>
                <span className="inspector-metric-val mono tabular-nums">
                  {selectedCoords[0].toFixed(5)}, {selectedCoords[1].toFixed(5)}
                </span>
              </div>
            </div>

            <div className="inspector-footer">
              <button
                type="button"
                className="inspector-action-btn primary"
                onClick={() => {
                  setViewTarget({
                    coords: selectedCoords,
                    mode: 'fly',
                    key: Date.now(),
                  })
                }}
              >
                <span className="btn-text-full">🔍 Focus on Point</span>
                <span className="btn-text-mobile">🔍 Focus</span>
              </button>
              <button
                type="button"
                className="inspector-action-btn secondary"
                onClick={handleCloseInspector}
              >
                <span className="btn-text-full">Return to Person</span>
                <span className="btn-text-mobile">Return</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {!activePosition && !isCaregiverMode && !isMultiPatientView && (
        <div className="map-overlay-notice">
          No recorded location data for this patient
        </div>
      )}

      {/* Timeline Section: Multi-Patient Overview OR Single Patient Journey Timeline */}
      {isMultiPatientView ? (
        <div className="movement-timeline-container" role="region" aria-label="Multi-Patient Trajectory Overview">
          <div className="timeline-header">
            <div className="timeline-title-wrap">
              <h3 className="timeline-title">👥 Multi-Patient Trajectory Comparison</h3>
              <span className="timeline-subtext">
                Comparing real recorded trajectories. Each user&apos;s data is fetched independently from their own backend history.
              </span>
            </div>
          </div>
          <div className="multi-patient-summary-grid">
            {multiPatientIds.map((pid, idx) => {
              const p = multiPatientData[pid]
              const color = p?.color || MULTI_PATIENT_PALETTE[idx % MULTI_PATIENT_PALETTE.length].color
              return (
                <div key={pid} className="multi-patient-summary-card" style={{ borderLeft: `4px solid ${color}` }}>
                  <div className="summary-card-header">
                    <span className="summary-patient-id" style={{ color }}>👤 Patient: {pid}</span>
                    <span className={`summary-status-tag ${p?.hasData ? 'has-data' : p?.exists ? 'no-data' : 'unknown'}`}>
                      {p?.loading
                        ? 'Loading...'
                        : p?.hasData
                        ? `${p.coords.length} Points Recorded`
                        : p?.exists
                        ? 'No Recorded Location Data'
                        : 'Unknown Patient ID (Not Found)'}
                    </span>
                  </div>
                  {p?.hasData ? (
                    <div className="summary-card-body">
                      <div>
                        <strong>First Waypoint:</strong> {formatTimestamp(p.trailMetadata[0]?.timestamp)}
                      </div>
                      <div>
                        <strong>Latest Waypoint:</strong> {formatTimestamp(p.trailMetadata[p.trailMetadata.length - 1]?.timestamp)}
                      </div>
                      <div className="mono" style={{ fontSize: '11px', color: '#64748b' }}>
                        Latest Lat/Lng: {p.coords[p.coords.length - 1][0].toFixed(5)}, {p.coords[p.coords.length - 1][1].toFixed(5)}
                      </div>
                    </div>
                  ) : (
                    <div className="summary-card-body">
                      <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
                        {p?.exists
                          ? 'Patient exists in database with 0 recorded location points.'
                          : 'Patient ID was not found in the database.'}
                      </p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        (isCaregiverMode || locationHistory.length > 0) && (
          <div
            className="movement-timeline-container"
            role="region"
            aria-label="Walking movement trail timeline"
          >
            <div className="timeline-header">
              <div className="timeline-title-wrap">
                <h3 className="timeline-title">🚶 Walking Trail Timeline</h3>
                <span className="timeline-subtext">
                  Chronological breadcrumbs ({locationHistory.length} recorded waypoints). Select any point to inspect and focus map.
                </span>
              </div>
              {selectedPointIndex !== null && (
                <button
                  type="button"
                  className="timeline-clear-btn"
                  onClick={handleCloseInspector}
                >
                  Clear Selection (#{selectedPointIndex + 1})
                </button>
              )}
            </div>

            {locationHistory.length === 0 ? (
              <div className="timeline-empty-state">
                <span className="timeline-empty-icon">{patientStatus && !patientStatus.exists ? '❓' : '👣'}</span>
                <div className="timeline-empty-text">
                  <span className="timeline-empty-title">No recorded location data for this patient</span>
                  <span className="timeline-empty-desc">
                    {patientStatus && !patientStatus.exists
                      ? `Unknown patient ID "${userId || 'unspecified'}". No registered patient record or consent found in the database.`
                      : `No recorded location data for this patient (${userId || 'unspecified'}). Chronological breadcrumbs will appear here as telemetry is recorded.`}
                  </span>
                </div>
              </div>
            ) : (
              <div
                className="journey-feed-list"
                ref={timelineRef}
                tabIndex={0}
                role="list"
                aria-label="Chronological movement waypoints"
              >
                <div className="journey-hairline" />
                {locationHistory.map((coords, idx) => {
                  const meta = trailMetadata[idx] || {}
                  const isSelected = selectedPointIndex === idx
                  const isLatest = idx === locationHistory.length - 1
                  const tier = meta.riskTier || 'QUIESCENT'
                  const tierColor = TIER_COLORS[tier] || '#64748b'
                  const tierLabel = TIER_LABELS[tier] || 'Low Activity'
                  const timeDisplay = formatTimestamp(meta.timestamp)
                  const relTime = formatRelativeTime(meta.timestamp, now)

                  return (
                    <div
                      key={`journey-item-${idx}`}
                      data-waypoint-index={idx}
                      className={`journey-feed-item journey-item-${idx} ${
                        isSelected ? 'journey-item-selected' : ''
                      } ${isLatest ? 'journey-item-latest' : ''}`}
                      role="button"
                      tabIndex={0}
                      aria-selected={isSelected}
                      aria-label={`Waypoint ${idx + 1} of ${
                        locationHistory.length
                      }, recorded ${timeDisplay}, status ${tierLabel}`}
                      onClick={() => handleSelectPoint(idx)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          handleSelectPoint(idx)
                        }
                      }}
                    >
                      <div
                        className={`journey-node-disc ${
                          isSelected ? 'node-selected' : ''
                        } ${isLatest ? 'node-latest' : ''}`}
                        style={{ borderColor: tierColor }}
                      >
                        <span className="journey-node-number">{idx + 1}</span>
                      </div>

                      <div className="journey-card">
                        <div className="journey-card-header">
                          <div className="journey-card-title-group">
                            <span className="journey-waypoint-name">
                              Waypoint #{idx + 1}
                            </span>
                            {isLatest && (
                              <span className="journey-tag-latest">
                                {isLive ? '● Live Current' : '● Last Known'}
                              </span>
                            )}
                          </div>
                          <div className="journey-card-time tabular-nums">
                            <span>{timeDisplay}</span>
                            {relTime && (
                              <span className="journey-time-rel">
                                {' '}· {relTime}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="journey-card-details">
                          <span
                            className="journey-tier-pill"
                            style={{
                              backgroundColor: `${tierColor}18`,
                              color: tierColor,
                              borderColor: `${tierColor}40`,
                            }}
                          >
                            {tierLabel}
                          </span>
                          <span className="journey-metric-pill tabular-nums">
                            ⚡ {meta.speed != null && !isNaN(meta.speed)
                              ? `${(Number(meta.speed) * 3.6).toFixed(1)} km/h`
                              : 'Stationary'}
                          </span>
                          <span className="journey-coords-text mono tabular-nums">
                            📍 {coords[0].toFixed(5)}, {coords[1].toFixed(5)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      )}
    </div>
  )
}
