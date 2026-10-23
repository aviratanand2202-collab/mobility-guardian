import React, { useState, useEffect } from 'react'

export default function ModelComparisonDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedHorizon, setSelectedHorizon] = useState('120s')
  const [viewVariant, setViewVariant] = useState('revised') // 'revised' (tuned) vs 'baseline'

  useEffect(() => {
    fetchComparisonData()
  }, [])

  const fetchComparisonData = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/models/comparison')
      if (!res.ok) {
        throw new Error(`Failed to fetch model comparison (status ${res.status})`)
      }
      const json = await res.json()
      setData(json)
    } catch (err) {
      console.error('Error fetching model comparison data:', err)
      setError(err.message || 'Network error fetching model comparison data.')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card" style={{ padding: '40px', textAlign: 'center' }}>
        <div style={{ fontSize: '18px', fontWeight: 600, color: '#374151', marginBottom: '8px' }}>
          Loading Model Benchmark Data...
        </div>
        <p style={{ color: '#6b7280', margin: 0 }}>
          Retrieving verified held-out test evaluation artifacts from backend.
        </p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card" style={{ padding: '32px', textAlign: 'center', borderLeft: '4px solid #ef4444' }}>
        <h3 style={{ color: '#b91c1c', margin: '0 0 10px 0' }}>Unable to Load Model Benchmarks</h3>
        <p style={{ color: '#4b5563', marginBottom: '16px' }}>{error || 'No evaluation data received.'}</p>
        <button type="button" className="btn btn-primary btn-sm" onClick={fetchComparisonData}>
          Retry Fetch
        </button>
      </div>
    )
  }

  const { models, test_evaluations, dataset_info, production_default } = data
  const horizons = data.horizons || ['120s', '360s', '600s', '840s']

  // Helper to extract metric for a model at the active horizon
  const getMetric = (modelId, metricKey) => {
    const horizonData = test_evaluations?.[modelId]?.[selectedHorizon]
    if (!horizonData) return null
    const variantData = viewVariant === 'revised' ? horizonData.revised : horizonData.baseline
    return variantData?.[metricKey]
  }

  const getConfusionMatrix = (modelId) => {
    const horizonData = test_evaluations?.[modelId]?.[selectedHorizon]
    if (!horizonData) return { tn: 0, fp: 0, fn: 0, tp: 0 }
    const variantData = viewVariant === 'revised' ? horizonData.revised : horizonData.baseline
    return variantData?.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 }
  }

  const getDelta = (modelId, deltaKey) => {
    return test_evaluations?.[modelId]?.[selectedHorizon]?.delta?.[deltaKey] ?? 0
  }

  const formatPct = (val) => (val != null ? `${(val * 100).toFixed(2)}%` : 'N/A')
  const formatDec = (val, d = 4) => (val != null ? Number(val).toFixed(d) : 'N/A')

  return (
    <div className="model-comparison-dashboard" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header & Context Bar */}
      <div className="card" style={{ background: '#ffffff', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: '#111827' }}>
                Predictive Model Family Comparison & Multi-Horizon Benchmarks
              </h2>
              <span
                style={{
                  background: '#dcfce7',
                  color: '#166534',
                  fontSize: '12px',
                  fontWeight: 700,
                  padding: '3px 10px',
                  borderRadius: '9999px',
                  border: '1px solid #bbf7d0',
                }}
              >
                Production Default: Improved XGBoost
              </span>
            </div>
            <p style={{ margin: '6px 0 0 0', color: '#6b7280', fontSize: '13px' }}>
              Independent comparative benchmarks across 72-feature schemas on the held-out test cohort (N = 16,697 - 76,022 windows).
            </p>
          </div>

          {/* Controls: Horizon Selector, Mode Toggle, and Refresh */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#4b5563' }}>Prediction Horizon:</span>
              <div style={{ display: 'inline-flex', borderRadius: '6px', border: '1px solid #d1d5db', overflow: 'hidden' }}>
                {horizons.map((h) => (
                  <button
                    key={h}
                    type="button"
                    onClick={() => setSelectedHorizon(h)}
                    style={{
                      padding: '6px 12px',
                      fontSize: '13px',
                      fontWeight: selectedHorizon === h ? 700 : 500,
                      background: selectedHorizon === h ? '#2563eb' : '#ffffff',
                      color: selectedHorizon === h ? '#ffffff' : '#374151',
                      border: 'none',
                      cursor: 'pointer',
                      borderRight: '1px solid #d1d5db',
                      transition: 'background-color 0.15s',
                    }}
                  >
                    {h}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#4b5563' }}>Thresholds:</span>
              <div style={{ display: 'inline-flex', borderRadius: '6px', border: '1px solid #d1d5db', overflow: 'hidden' }}>
                <button
                  type="button"
                  onClick={() => setViewVariant('revised')}
                  style={{
                    padding: '6px 12px',
                    fontSize: '13px',
                    fontWeight: viewVariant === 'revised' ? 700 : 500,
                    background: viewVariant === 'revised' ? '#059669' : '#ffffff',
                    color: viewVariant === 'revised' ? '#ffffff' : '#374151',
                    border: 'none',
                    cursor: 'pointer',
                    borderRight: '1px solid #d1d5db',
                    transition: 'background-color 0.15s',
                  }}
                >
                  FP-Tuned (Active)
                </button>
                <button
                  type="button"
                  onClick={() => setViewVariant('baseline')}
                  style={{
                    padding: '6px 12px',
                    fontSize: '13px',
                    fontWeight: viewVariant === 'baseline' ? 700 : 500,
                    background: viewVariant === 'baseline' ? '#6b7280' : '#ffffff',
                    color: viewVariant === 'baseline' ? '#ffffff' : '#374151',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                  }}
                >
                  Macro-F1 Baseline
                </button>
              </div>
            </div>

            <button type="button" className="btn btn-secondary btn-sm" onClick={fetchComparisonData}>
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Model Family Architecture Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
        {models.map((m) => {
          const isProd = m.id === production_default
          const balAcc = getMetric(m.id, 'balanced_accuracy')
          const fpr = getMetric(m.id, 'false_positive_rate')
          const recall = getMetric(m.id, 'recall')
          const thresh = getMetric(m.id, 'threshold')

          return (
            <div
              key={m.id}
              className="card"
              style={{
                background: '#ffffff',
                border: isProd ? '2px solid #2563eb' : '1px solid #e5e7eb',
                borderRadius: '8px',
                padding: '16px',
                position: 'relative',
                boxShadow: isProd ? '0 4px 6px -1px rgba(37, 99, 235, 0.1)' : 'none',
              }}
            >
              {isProd && (
                <div
                  style={{
                    position: 'absolute',
                    top: '-10px',
                    right: '16px',
                    background: '#2563eb',
                    color: '#ffffff',
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '4px',
                    letterSpacing: '0.05em',
                  }}
                >
                  ACTIVE PRODUCTION
                </div>
              )}
              <h3 style={{ margin: '0 0 6px 0', fontSize: '16px', fontWeight: 700, color: '#111827' }}>
                {m.name}
              </h3>
              <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#6b7280', minHeight: '36px' }}>
                {m.description}
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', background: '#f9fafb', padding: '10px', borderRadius: '6px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: '#6b7280', textTransform: 'uppercase' }}>Balanced Acc</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#111827' }}>{formatPct(balAcc)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#6b7280', textTransform: 'uppercase' }}>False Pos Rate</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#dc2626' }}>{formatPct(fpr)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#6b7280', textTransform: 'uppercase' }}>True Positive Rate</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#059669' }}>{formatPct(recall)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#6b7280', textTransform: 'uppercase' }}>Threshold</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#4b5563' }}>t = {formatDec(thresh, 2)}</div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Visual Comparison Charts (FPR and Balanced Accuracy) */}
      <div className="card" style={{ background: '#ffffff', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
        <h3 style={{ margin: '0 0 14px 0', fontSize: '16px', fontWeight: 700, color: '#111827' }}>
          Comparative Visual Metrics ({selectedHorizon} Horizon, {viewVariant === 'revised' ? 'FP-Tuned' : 'Baseline'} Mode)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
          {/* FPR Bar Comparison */}
          <div style={{ background: '#f9fafb', padding: '14px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#374151', marginBottom: '12px' }}>
              False Positive Rate (FPR) — Lower is Better
            </div>
            {models.map((m) => {
              const fprVal = getMetric(m.id, 'false_positive_rate') || 0
              const pct = (fprVal * 100).toFixed(2)
              const isProd = m.id === production_default
              return (
                <div key={m.id} style={{ marginBottom: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
                    <span style={{ fontWeight: isProd ? 700 : 500, color: isProd ? '#1d4ed8' : '#374151' }}>
                      {m.name} {isProd && '(Prod)'}
                    </span>
                    <span style={{ fontWeight: 700, color: '#dc2626' }}>{pct}%</span>
                  </div>
                  <div style={{ height: '10px', background: '#e5e7eb', borderRadius: '5px', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, Math.max(0, fprVal * 100))}%`,
                        background: isProd ? '#dc2626' : '#f87171',
                        borderRadius: '5px',
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Balanced Accuracy Bar Comparison */}
          <div style={{ background: '#f9fafb', padding: '14px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#374151', marginBottom: '12px' }}>
              Balanced Accuracy (TPR + TNR)/2 — Higher is Better
            </div>
            {models.map((m) => {
              const balVal = getMetric(m.id, 'balanced_accuracy') || 0
              const pct = (balVal * 100).toFixed(2)
              const isProd = m.id === production_default
              return (
                <div key={m.id} style={{ marginBottom: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
                    <span style={{ fontWeight: isProd ? 700 : 500, color: isProd ? '#1d4ed8' : '#374151' }}>
                      {m.name} {isProd && '(Prod)'}
                    </span>
                    <span style={{ fontWeight: 700, color: '#059669' }}>{pct}%</span>
                  </div>
                  <div style={{ height: '10px', background: '#e5e7eb', borderRadius: '5px', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, Math.max(0, balVal * 100))}%`,
                        background: isProd ? '#2563eb' : '#3b82f6',
                        borderRadius: '5px',
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Main Comparative Metrics Benchmark Table */}
      <div className="card" style={{ background: '#ffffff', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
        <h3 style={{ margin: '0 0 14px 0', fontSize: '16px', fontWeight: 700, color: '#111827' }}>
          Held-Out Test Metrics Benchmark ({selectedHorizon} Horizon, {viewVariant === 'revised' ? 'FP-Controlled' : 'Baseline'} Mode)
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ padding: '10px 12px', fontWeight: 700, color: '#374151' }}>Metric</th>
                <th style={{ padding: '10px 12px', fontWeight: 700, color: '#1d4ed8' }}>Improved XGBoost (Prod)</th>
                <th style={{ padding: '10px 12px', fontWeight: 700, color: '#15803d' }}>Random Forest</th>
                <th style={{ padding: '10px 12px', fontWeight: 700, color: '#c2410c' }}>LightGBM</th>
                <th style={{ padding: '10px 12px', fontWeight: 700, color: '#4b5563' }}>Operational Interpretation / Formula</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Operating Threshold (t)</td>
                <td style={{ padding: '10px 12px', fontWeight: 700, color: '#1d4ed8' }}>t = {formatDec(getMetric('xgboost_improved', 'threshold'), 2)}</td>
                <td style={{ padding: '10px 12px' }}>t = {formatDec(getMetric('random_forest', 'threshold'), 2)}</td>
                <td style={{ padding: '10px 12px' }}>t = {formatDec(getMetric('lightgbm', 'threshold'), 2)}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>Decision boundary on calibrated probability space</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6', background: '#f9fafb' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>False Positive Rate (FPR)</td>
                <td style={{ padding: '10px 12px', fontWeight: 700, color: '#b91c1c' }}>{formatPct(getMetric('xgboost_improved', 'false_positive_rate'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('random_forest', 'false_positive_rate'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('lightgbm', 'false_positive_rate'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>FP / (FP + TN) — Lower is better; fraction of routine windows generating false alerts</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Balanced Accuracy</td>
                <td style={{ padding: '10px 12px', fontWeight: 700, color: '#1d4ed8' }}>{formatPct(getMetric('xgboost_improved', 'balanced_accuracy'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('random_forest', 'balanced_accuracy'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('lightgbm', 'balanced_accuracy'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>(TPR + TNR)/2 — Equal weight to anomaly detection and routine protection</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6', background: '#f9fafb' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Recall (Sensitivity)</td>
                <td style={{ padding: '10px 12px', fontWeight: 600, color: '#059669' }}>{formatPct(getMetric('xgboost_improved', 'recall'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('random_forest', 'recall'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('lightgbm', 'recall'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>TP / (TP + FN) — Fraction of kinematic outliers successfully flagged</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Specificity</td>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>{formatPct(getMetric('xgboost_improved', 'specificity'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('random_forest', 'specificity'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('lightgbm', 'specificity'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>TN / (FP + TN) — 1 - FPR; fraction of routine movements correctly muted</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6', background: '#f9fafb' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Precision (Positive Predictive Value)</td>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>{formatPct(getMetric('xgboost_improved', 'precision'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('random_forest', 'precision'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatPct(getMetric('lightgbm', 'precision'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>TP / (TP + FP) — Probability that an active alert is a true outlier</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Macro-F1 Score</td>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>{formatDec(getMetric('xgboost_improved', 'macro_f1'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('random_forest', 'macro_f1'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('lightgbm', 'macro_f1'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>Unweighted average of positive and negative F1</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6', background: '#f9fafb' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>PR-AUC</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('xgboost_improved', 'pr_auc'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('random_forest', 'pr_auc'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('lightgbm', 'pr_auc'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>Precision-Recall Area Under Curve across all thresholds</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>ROC-AUC</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('xgboost_improved', 'roc_auc'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('random_forest', 'roc_auc'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('lightgbm', 'roc_auc'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>Receiver Operating Characteristic AUC</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f3f4f6', background: '#f9fafb' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Brier Calibration Score</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('xgboost_improved', 'brier_score'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('random_forest', 'brier_score'))}</td>
                <td style={{ padding: '10px 12px' }}>{formatDec(getMetric('lightgbm', 'brier_score'))}</td>
                <td style={{ padding: '10px 12px', color: '#6b7280', fontSize: '12px' }}>Mean squared probability calibration error (lower is better)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrices Side-by-Side */}
      <div className="card" style={{ background: '#ffffff', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
        <h3 style={{ margin: '0 0 6px 0', fontSize: '16px', fontWeight: 700, color: '#111827' }}>
          Real Confusion Matrices ({selectedHorizon} Horizon, {viewVariant === 'revised' ? 'FP-Controlled' : 'Baseline'} Mode)
        </h3>
        <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#6b7280' }}>
          Exact sample classification breakdown on the held-out test cohort.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          {models.map((m) => {
            const cm = getConfusionMatrix(m.id)
            const dFp = getDelta(m.id, 'delta_fp')
            const dFn = getDelta(m.id, 'delta_fn')
            const isProd = m.id === production_default

            return (
              <div
                key={m.id}
                style={{
                  background: '#f9fafb',
                  border: isProd ? '1px solid #2563eb' : '1px solid #e5e7eb',
                  borderRadius: '6px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontWeight: 700, fontSize: '14px', color: '#111827' }}>{m.name}</span>
                  {viewVariant === 'revised' && (
                    <span style={{ fontSize: '12px', fontWeight: 700, color: dFp < 0 ? '#15803d' : '#4b5563' }}>
                      FP Reduction: {dFp}
                    </span>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', textAlign: 'center' }}>
                  {/* True Negative */}
                  <div style={{ background: '#eff6ff', padding: '10px', borderRadius: '4px', border: '1px solid #bfdbfe' }}>
                    <div style={{ fontSize: '11px', color: '#1e40af', fontWeight: 600 }}>True Negative (TN)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#1e3a8a' }}>{cm.tn.toLocaleString()}</div>
                    <div style={{ fontSize: '10px', color: '#2563eb', marginTop: '2px', lineHeight: '1.2' }}>
                      Routine-Like Window Correctly Classified as Negative
                    </div>
                  </div>

                  {/* False Positive */}
                  <div style={{ background: '#fef2f2', padding: '10px', borderRadius: '4px', border: '1px solid #fecaca' }}>
                    <div style={{ fontSize: '11px', color: '#991b1b', fontWeight: 600 }}>False Positive (FP)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#7f1d1d' }}>{cm.fp.toLocaleString()}</div>
                    <div style={{ fontSize: '10px', color: '#dc2626', marginTop: '2px', lineHeight: '1.2' }}>
                      Routine-Like Window Classified as Positive (False Alert)
                    </div>
                  </div>

                  {/* False Negative */}
                  <div style={{ background: '#fef3c7', padding: '10px', borderRadius: '4px', border: '1px solid #fde68a' }}>
                    <div style={{ fontSize: '11px', color: '#92400e', fontWeight: 600 }}>False Negative (FN)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#78350f' }}>{cm.fn.toLocaleString()}</div>
                    <div style={{ fontSize: '10px', color: '#d97706', marginTop: '2px', lineHeight: '1.2' }}>
                      Kinematic Outlier Classified as Negative
                    </div>
                  </div>

                  {/* True Positive */}
                  <div style={{ background: '#f0fdf4', padding: '10px', borderRadius: '4px', border: '1px solid #bbf7d0' }}>
                    <div style={{ fontSize: '11px', color: '#166534', fontWeight: 600 }}>True Positive (TP)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#14532d' }}>{cm.tp.toLocaleString()}</div>
                    <div style={{ fontSize: '10px', color: '#16a34a', marginTop: '2px', lineHeight: '1.2' }}>
                      Kinematic Outlier Correctly Classified as Positive
                    </div>
                  </div>
                </div>

                {viewVariant === 'revised' && (
                  <div style={{ marginTop: '10px', fontSize: '11px', color: '#4b5563', textAlign: 'center' }}>
                    Net Delta vs Baseline: <strong style={{ color: '#15803d' }}>{dFp} false alarms</strong> eliminated;{' '}
                    <strong style={{ color: '#b45309' }}>+{dFn} kinematic outliers classified as negative</strong>.
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Dataset & Non-Clinical Disclaimer */}
      <div
        className="card"
        style={{
          background: '#fffbeb',
          border: '1px solid #fde68a',
          padding: '16px 20px',
          borderRadius: '8px',
          fontSize: '12px',
          color: '#92400e',
        }}
      >
        <div style={{ fontWeight: 700, fontSize: '13px', marginBottom: '4px' }}>
          Scientific Evaluation Parameters & Operational Disclaimer
        </div>
        <p style={{ margin: '0 0 6px 0' }}>
          <strong>Cohort Split:</strong> {dataset_info?.evaluation_split} ({dataset_info?.sample_counts?.[selectedHorizon]?.toLocaleString()} samples at {selectedHorizon}). Trajectories were partitioned strictly chronologically per patient with zero cross-split trajectory overlap.
        </p>
        <p style={{ margin: '0 0 6px 0' }}>
          <strong>Target Definition:</strong> {dataset_info?.target_definition}.
        </p>
        <p style={{ margin: 0 }}>
          <strong>Disclaimer:</strong> {dataset_info?.disclaimer}
        </p>
      </div>
    </div>
  )
}
