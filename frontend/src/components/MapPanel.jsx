import React, { useState, useEffect, useRef } from 'react'
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  CircleMarker,
  Tooltip,
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

// Custom DivIcon creator inspired by Reference 2 (vibrant pin with halo)
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
}) {
  const [selectedPointIndex, setSelectedPointIndex] = useState(null)
  const [viewTarget, setViewTarget] = useState(null)
  const [now, setNow] = useState(Date.now())
  const timelineRef = useRef(null)

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

  // Fallback anchor (e.g. San Francisco) when no coordinates yet
  const defaultCenter = [37.7749, -122.4194]
  const activePosition =
    currentLocation && currentLocation.lat != null && currentLocation.lng != null
      ? [currentLocation.lat, currentLocation.lng]
      : locationHistory.length > 0
      ? locationHistory[locationHistory.length - 1]
      : null

  const center = activePosition || defaultCenter
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

  // Recenter handler (Requirement 1)
  const handleRecenter = () => {
    if (activePosition) {
      setSelectedPointIndex(null)
      setViewTarget({ coords: activePosition, mode: 'fly', key: Date.now() })
    }
  }

  // Focus Trail handler (Section 6 approved action)
  const handleFocusTrail = () => {
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

  // Select trail point handler (Requirement 2, 3, 5)
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
            {isCaregiverMode
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
          {isCaregiverMode ? (
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
            activePosition && (
              <span className="mono" style={{ fontSize: '11px', color: '#6b7280' }}>
                ({activePosition[0].toFixed(5)}, {activePosition[1].toFixed(5)})
              </span>
            )
          )}
        </div>

        <div className="map-header-toolbar">
          {isCaregiverMode && (
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
          )}

          <span className="badge-count tabular-nums">
            {isCaregiverMode
              ? isLive && activePosition
                ? `${locationHistory.length} trail point${locationHistory.length === 1 ? '' : 's'}`
                : `${locationHistory.length} point${locationHistory.length === 1 ? '' : 's'}${isTelemetryStale ? ' (stale)' : hasHistoricalOnly ? ' (cached)' : ' (historical)'}`
              : `${locationHistory.length} trail point${locationHistory.length === 1 ? '' : 's'}`}
          </span>
        </div>
      </div>

      {/* Map Canvas Wrapper */}
      <div className="map-wrapper">
        {isCaregiverMode && !activePosition ? (
          <div className={`map-empty-telemetry-container ${
            riskTier === 'CRITICAL'
              ? 'map-empty-critical'
              : riskTier === 'SUSPICIOUS'
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
                  {riskTier === 'CRITICAL' ? '🚨' : riskTier === 'SUSPICIOUS' ? '⚠️' : '🛰️'}
                </span>
              </div>
            </div>
            <div className="map-empty-telemetry-content">
              {riskTier === 'CRITICAL' ? (
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
              ) : riskTier === 'SUSPICIOUS' ? (
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
                  <span className="map-empty-pill">Awaiting GPS Telemetry</span>
                  <h3 className="map-empty-title">No Location Coordinates Yet</h3>
                  <p className="map-empty-desc">
                    {isConnecting
                      ? 'Connecting to stream. Establishing connection to satellite telemetry feed...'
                      : isLive || !isOffline
                      ? 'Connected to stream. Waiting for the monitored device to transmit its initial satellite location fix.'
                      : 'Monitoring stream is currently offline. Connect to begin receiving location updates.'}
                  </p>
                  <div className="map-empty-tags">
                    <span className="empty-tag">User: <strong>{personLabel}</strong></span>
                    <span className="empty-tag">Status: <strong>{isOffline ? 'Offline' : isConnecting ? 'Connecting' : 'Standby for Fix'}</strong></span>
                    <span className="empty-tag">Trail: <strong>0 points</strong></span>
                  </div>
                </>
              )}
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
              activePosition={activePosition}
              isInspecting={selectedPointIndex !== null}
            />

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

            {/* Selectable Trail CircleMarkers (Caregiver Mode) */}
            {isCaregiverMode &&
              locationHistory.map((coords, idx) => {
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
          </MapContainer>
        )}

        {/* Truthful Corner Calibration Badge (matching Stitch corner overlay) */}
        {isCaregiverMode && activePosition && (
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

        {/* Floating Controls Overlay (Reference 2 style) */}
        {isCaregiverMode && activePosition && (
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

        {/* Contextual Point-Inspector Card (Reference 2 & Stitch Pod design) */}
        {isCaregiverMode && selectedCoords && (
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

      {!activePosition && !isCaregiverMode && (
        <div className="map-overlay-notice">
          Waiting for live telemetry location points over /ws/risk...
        </div>
      )}

      {/* Chronological Walking Journey Timeline / Feed */}
      {isCaregiverMode && (
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
              <span className="timeline-empty-icon">👣</span>
              <div className="timeline-empty-text">
                <span className="timeline-empty-title">Walking Trail Standby</span>
                <span className="timeline-empty-desc">
                  Chronological breadcrumbs will appear here sequentially as the device reports locations.
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
      )}
    </div>
  )
}
