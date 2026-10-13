import React from 'react'
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
  activeRisk,
  latestRisk,
  historyEntries,
  alertMessages,
  handleAlertUpdate,
  historyLoading,
  historyError,
  fetchHistory,
}) {
  return (
    <div className="technical-dashboard-container">
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
            currentRiskTier={activeRisk?.risk_tier}
            currentRiskScore={activeRisk?.risk_score}
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
            {isConnected
              ? 'No history found for this user in the database.'
              : 'Connect to query database history for this user.'}
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
