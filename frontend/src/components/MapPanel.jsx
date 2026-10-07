import React, { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'

// Color map for risk tiers per spec
const TIER_COLORS = {
  QUIESCENT: '#2563eb', // Blue
  NORMAL_TRANSIT: '#16a34a', // Green
  SUSPICIOUS: '#ea580c', // Orange
  CRITICAL: '#dc2626', // Red
}

// Custom DivIcon creator to avoid broken default Leaflet png asset URLs
function createRiskMarkerIcon(riskTier) {
  const color = TIER_COLORS[riskTier] || '#6b7280'
  const html = `
    <div style="position: relative; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;">
      <div style="
        position: absolute;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background-color: ${color};
        opacity: 0.3;
        animation: pulse 1.8s infinite;
      "></div>
      <div style="
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: ${color};
        border: 2px solid #ffffff;
        box-shadow: 0 0 6px rgba(0,0,0,0.4);
      "></div>
    </div>
  `
  return L.divIcon({
    className: 'custom-risk-marker',
    html,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -14],
  })
}

// Controller component to smoothly pan/center map when latest position changes
function MapCenterController({ position }) {
  const map = useMap()
  useEffect(() => {
    if (position && position[0] && position[1]) {
      map.panTo(position, { animate: true, duration: 0.5 })
    }
  }, [map, position])
  return null
}

export default function MapPanel({ currentLocation, locationHistory, currentRiskTier, currentRiskScore, userId }) {
  // Fallback anchor (e.g. San Francisco or default center) when no coordinates yet
  const defaultCenter = [37.7749, -122.4194]
  const activePosition = currentLocation && currentLocation.lat && currentLocation.lng
    ? [currentLocation.lat, currentLocation.lng]
    : null

  const center = activePosition || defaultCenter
  const riskTier = currentRiskTier || 'QUIESCENT'
  const markerIcon = createRiskMarkerIcon(riskTier)
  const trailColor = TIER_COLORS[riskTier] || '#2563eb'

  return (
    <div className="panel map-panel-container">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h2 className="panel-title">Spatial Tracking &amp; Trajectory Map</h2>
          {activePosition && (
            <span className="mono" style={{ fontSize: '11px', color: '#6b7280' }}>
              ({activePosition[0].toFixed(5)}, {activePosition[1].toFixed(5)})
            </span>
          )}
        </div>
        <span className="badge-count">
          {locationHistory.length} trail point{locationHistory.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="map-wrapper" style={{ height: '480px', width: '100%', borderRadius: '6px', overflow: 'hidden' }}>
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

          {activePosition && <MapCenterController position={activePosition} />}

          {/* Faint trail polyline of last ~30 locations */}
          {locationHistory.length > 1 && (
            <Polyline
              positions={locationHistory}
              pathOptions={{
                color: trailColor,
                weight: 4,
                opacity: 0.6,
                dashArray: '2, 6',
                lineJoin: 'round',
              }}
            />
          )}

          {/* Active user location marker */}
          {activePosition && (
            <Marker position={activePosition} icon={markerIcon}>
              <Popup>
                <div style={{ fontSize: '12px', lineHeight: 1.4 }}>
                  <strong>User:</strong> {userId}<br />
                  <strong>Risk Tier:</strong> {riskTier}<br />
                  <strong>Score:</strong> {currentRiskScore != null ? Number(currentRiskScore).toFixed(1) : '-'}<br />
                  <strong>Lat/Lng:</strong> {activePosition[0].toFixed(5)}, {activePosition[1].toFixed(5)}
                </div>
              </Popup>
            </Marker>
          )}
        </MapContainer>
      </div>

      {!activePosition && (
        <div className="map-overlay-notice">
          Waiting for live telemetry location points over /ws/risk...
        </div>
      )}
    </div>
  )
}
