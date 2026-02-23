import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Editor from '@monaco-editor/react';
import './DSLBuilderPage.css';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type DSLConfig = {
  data?: {
    source: string;
    calendar: string;
    start_date: string;
    end_date: string;
  };
  universe?: {
    name: string;
    filters: string[];
  };
  factors?: Record<string, { type: string; params: Record<string, any> }>;
  signals?: Record<string, { type: string; params: Record<string, any> }>;
  portfolio?: {
    type: 'long_short' | 'long_only';
    long_book?: { selector: string; weighting: string };
    short_book?: { selector: string; weighting: string };
  };
};

type DSLCode = {
  python_code: string;
  json_config: string;
};

type StrategyInfo = {
  id: string;
  name?: string;
  path?: string;
  description?: string;
};

type EditorMode = 'generated' | 'free_edit';

/* ------------------------------------------------------------------ */
/*  Factor / Signal type catalogues                                    */
/* ------------------------------------------------------------------ */

const FACTOR_TYPES = [
  { value: 'momentum', label: 'Momentum (ReturnFactor)' },
  { value: 'volatility', label: 'Volatility (VolatilityFactor)' },
  { value: 'overnight_return', label: 'Overnight Return' },
  { value: 'intraday_return', label: 'Intraday Return' },
  { value: 'fibo_retrace', label: 'Fibonacci Retrace' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
] as const;

const SIGNAL_TYPES = [
  { value: 'cross_section_rank', label: 'Cross-Section Rank' },
  { value: 'quantile', label: 'Quantile' },
  { value: 'mask_from_boolean', label: 'Mask From Boolean' },
  { value: 'less_equal', label: 'Less Equal' },
  { value: 'greater_equal', label: 'Greater Equal' },
  { value: 'ewm_mean', label: 'EWM Mean' },
  { value: 'rolling_mean', label: 'Rolling Mean' },
  { value: 'rolling_std', label: 'Rolling Std' },
  { value: 'diff', label: 'Diff' },
  { value: 'pct_change', label: 'Pct Change' },
] as const;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

async function fetchDSLCode(config: DSLConfig): Promise<DSLCode> {
  const res = await fetch('/api/dsl/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return (await res.json()) as DSLCode;
}

async function fetchStrategies(): Promise<StrategyInfo[]> {
  try {
    const res = await fetch('/api/strategies', { headers: { Accept: 'application/json' } });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data?.strategies) ? data.strategies : [];
  } catch {
    return [];
  }
}

async function fetchStrategySource(id: string): Promise<string> {
  const res = await fetch(`/api/strategies/${encodeURIComponent(id)}`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`Failed to load strategy: ${res.status}`);
  const data = await res.json();
  return String(data?.strategy?.source || data?.source || '');
}

async function saveStrategy(id: string, source: string, create: boolean): Promise<{ id: string; hash?: string }> {
  const url = create ? '/api/strategies' : `/api/strategies/${encodeURIComponent(id)}`;
  const method = create ? 'POST' : 'PUT';
  const body = create ? { id, source } : { source };
  const res = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const msg = data?.detail || data?.error?.message || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return res.json();
}

async function hashSource(source: string): Promise<string> {
  // SHA-256 via SubtleCrypto (available in all modern browsers).
  try {
    const enc = new TextEncoder().encode(source);
    const buf = await crypto.subtle.digest('SHA-256', enc);
    return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, '0')).join('');
  } catch {
    // Fallback: simple hash for environments without SubtleCrypto.
    return `fallback_${source.length}_${Date.now()}`;
  }
}

async function submitRun(strategyId: string, source: string): Promise<{ run_id: string; status: string }> {
  const hash = await hashSource(source);
  const res = await fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ strategy_id: strategyId, source, strategy_hash: hash }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const msg = data?.detail || data?.error?.message || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return res.json();
}

/* ------------------------------------------------------------------ */
/*  Default config                                                     */
/* ------------------------------------------------------------------ */

function defaultConfig(): DSLConfig {
  return {
    data: {
      source: 'parquet://equities/indicies.parquet',
      calendar: 'XNYS',
      start_date: '2015-01-01',
      end_date: '2025-12-31',
    },
    universe: {
      name: 'Indices',
      filters: [],
    },
    factors: {
      mom_126: { type: 'momentum', params: { lookback: 126 } },
    },
    signals: {
      rank_momentum: { type: 'cross_section_rank', params: { factor: 'mom_126' } },
    },
    portfolio: {
      type: 'long_short',
      long_book: { selector: 'TopN', weighting: 'EqualWeight' },
      short_book: { selector: 'BottomN', weighting: 'EqualWeight' },
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function DSLBuilderPage() {
  const [config, setConfig] = useState<DSLConfig>(defaultConfig);
  const [editorCode, setEditorCode] = useState<string>('');
  const [mode, setMode] = useState<EditorMode>('generated');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);

  // Strategy selector
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('');
  const [isNewStrategy, setIsNewStrategy] = useState(true);
  const [newStrategyId, setNewStrategyId] = useState('custom_strategy');

  const toastTimer = useRef<any>(null);

  // Strategy ID to use for save/run
  const activeStrategyId = useMemo(() => {
    if (!isNewStrategy && selectedStrategyId) return selectedStrategyId;
    return newStrategyId.trim() || 'custom_strategy';
  }, [isNewStrategy, selectedStrategyId, newStrategyId]);

  // Show toast and auto-clear
  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);

  // Load strategies list
  useEffect(() => {
    fetchStrategies().then(setStrategies);
  }, []);

  // Generate code from config (only in generated mode)
  useEffect(() => {
    if (mode !== 'generated') return;
    setLoading(true);
    setError('');
    fetchDSLCode(config)
      .then((code) => {
        setEditorCode(code.python_code);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [config, mode]);

  // Config update helpers
  const updateDataConfig = useCallback((key: string, value: string) => {
    setConfig((prev) => ({
      ...prev,
      data: { ...prev.data!, [key]: value },
    }));
  }, []);

  const updateUniverseName = useCallback((name: string) => {
    setConfig((prev) => ({
      ...prev,
      universe: { ...prev.universe!, name },
    }));
  }, []);

  const updateFactor = useCallback((name: string, type: string) => {
    setConfig((prev) => ({
      ...prev,
      factors: {
        ...prev.factors!,
        [name]: { type, params: prev.factors?.[name]?.params || { lookback: 126 } },
      },
    }));
  }, []);

  const addFactor = useCallback(() => {
    setConfig((prev) => {
      const existing = Object.keys(prev.factors || {});
      const n = existing.length + 1;
      const name = `factor_${n}`;
      return {
        ...prev,
        factors: {
          ...prev.factors,
          [name]: { type: 'momentum', params: { lookback: 126 } },
        },
      };
    });
  }, []);

  const removeFactor = useCallback((name: string) => {
    setConfig((prev) => {
      const next = { ...prev.factors };
      delete next[name];
      return { ...prev, factors: next };
    });
  }, []);

  const updateSignal = useCallback((name: string, type: string) => {
    setConfig((prev) => ({
      ...prev,
      signals: {
        ...prev.signals!,
        [name]: { type, params: prev.signals?.[name]?.params || { factor: 'mom_126' } },
      },
    }));
  }, []);

  const addSignal = useCallback(() => {
    setConfig((prev) => {
      const existing = Object.keys(prev.signals || {});
      const n = existing.length + 1;
      const name = `signal_${n}`;
      // Default: reference first factor if available
      const firstFactor = Object.keys(prev.factors || {})[0] || 'mom_126';
      return {
        ...prev,
        signals: {
          ...prev.signals,
          [name]: { type: 'cross_section_rank', params: { factor: firstFactor } },
        },
      };
    });
  }, []);

  const removeSignal = useCallback((name: string) => {
    setConfig((prev) => {
      const next = { ...prev.signals };
      delete next[name];
      return { ...prev, signals: next };
    });
  }, []);

  const updatePortfolioType = useCallback((type: 'long_short' | 'long_only') => {
    setConfig((prev) => ({
      ...prev,
      portfolio: { ...prev.portfolio!, type },
    }));
  }, []);

  // Load strategy source into editor
  const loadStrategy = useCallback(
    async (id: string) => {
      if (!id) return;
      try {
        const source = await fetchStrategySource(id);
        setEditorCode(source);
        setSelectedStrategyId(id);
        setIsNewStrategy(false);
        setMode('free_edit');
        showToast(`Loaded strategy: ${id}`, 'success');
      } catch (e: any) {
        showToast(`Failed to load: ${e.message}`, 'error');
      }
    },
    [showToast]
  );

  // Save action
  const handleSave = useCallback(async (): Promise<boolean> => {
    setSaving(true);
    try {
      await saveStrategy(activeStrategyId, editorCode, isNewStrategy);
      setIsNewStrategy(false);
      setSelectedStrategyId(activeStrategyId);
      showToast(`Strategy saved: ${activeStrategyId}`, 'success');
      // Refresh strategies list
      fetchStrategies().then(setStrategies);
      return true;
    } catch (e: any) {
      showToast(`Save failed: ${e.message}`, 'error');
      return false;
    } finally {
      setSaving(false);
    }
  }, [activeStrategyId, editorCode, isNewStrategy, showToast]);

  // Run action
  const handleRun = useCallback(async () => {
    setRunning(true);
    try {
      const result = await submitRun(activeStrategyId, editorCode);
      showToast(`Run submitted: ${result.run_id?.slice(0, 8)}`, 'success');
      // Navigate to Runs tab
      try {
        const u = new URL(window.location.href);
        u.searchParams.set('tab', 'runs');
        u.searchParams.set('run_id', result.run_id);
        window.history.replaceState({}, '', u.toString());
        (window as any)?.workspaceApi?.setTab?.('runs');
      } catch {}
    } catch (e: any) {
      showToast(`Run failed: ${e.message}`, 'error');
    } finally {
      setRunning(false);
    }
  }, [activeStrategyId, editorCode, showToast]);

  // Save and Run
  const handleSaveAndRun = useCallback(async () => {
    const ok = await handleSave();
    if (ok) await handleRun();
  }, [handleSave, handleRun]);

  // Toggle mode
  const toggleMode = useCallback(() => {
    setMode((prev) => (prev === 'generated' ? 'free_edit' : 'generated'));
  }, []);

  const isBusy = saving || running;
  const factorEntries = Object.entries(config.factors || {});
  const signalEntries = Object.entries(config.signals || {});

  return (
    <div className="dsl-builder" data-testid="dsl-builder-page">
      <div className="dsl-container">
        {/* Left Panel: Controls */}
        <div className={`dsl-controls ${mode === 'free_edit' ? 'dsl-controls--disabled' : ''}`}>
          <h2>DSL Strategy Builder</h2>
          {mode === 'free_edit' && (
            <div className="dsl-mode-notice" data-testid="dsl-free-edit-notice">
              Form is read-only in free-edit mode. Switch to Generated mode to use the form.
            </div>
          )}

          {/* Data Config Section */}
          <section className="config-section">
            <h3>Data Configuration</h3>
            <div className="form-group">
              <label>Data Source</label>
              <input
                type="text"
                value={config.data?.source || ''}
                onChange={(e) => updateDataConfig('source', e.target.value)}
                placeholder="parquet://equities/sp500_daily"
                disabled={mode === 'free_edit'}
              />
            </div>
            <div className="form-group">
              <label>Calendar</label>
              <select
                value={config.data?.calendar || 'XNYS'}
                onChange={(e) => updateDataConfig('calendar', e.target.value)}
                disabled={mode === 'free_edit'}
              >
                <option value="XNYS">XNYS (NYSE)</option>
                <option value="XETRA">XETRA (Deutsche Boerse)</option>
                <option value="XLON">XLON (LSE)</option>
              </select>
            </div>
            <div className="form-group">
              <label>Start Date</label>
              <input
                type="date"
                value={config.data?.start_date || '2015-01-01'}
                onChange={(e) => updateDataConfig('start_date', e.target.value)}
                disabled={mode === 'free_edit'}
              />
            </div>
            <div className="form-group">
              <label>End Date</label>
              <input
                type="date"
                value={config.data?.end_date || '2025-12-31'}
                onChange={(e) => updateDataConfig('end_date', e.target.value)}
                disabled={mode === 'free_edit'}
              />
            </div>
          </section>

          {/* Universe Section */}
          <section className="config-section">
            <h3>Universe</h3>
            <div className="form-group">
              <label>Universe Name</label>
              <input
                type="text"
                value={config.universe?.name || ''}
                onChange={(e) => updateUniverseName(e.target.value)}
                placeholder="e.g., SP500, Indices"
                disabled={mode === 'free_edit'}
              />
            </div>
          </section>

          {/* Factors Section */}
          <section className="config-section">
            <h3>
              Factors
              <button
                className="dsl-add-btn"
                data-testid="btnAddFactor"
                onClick={addFactor}
                disabled={mode === 'free_edit'}
                title="Add Factor"
              >
                + Add
              </button>
            </h3>
            {factorEntries.map(([name, factor]) => (
              <div className="form-group form-group-row" key={name}>
                <input
                  type="text"
                  value={name}
                  className="form-name-input"
                  readOnly
                  title={`Factor name: ${name}`}
                />
                <select
                  value={factor.type}
                  onChange={(e) => updateFactor(name, e.target.value)}
                  disabled={mode === 'free_edit'}
                >
                  {FACTOR_TYPES.map((ft) => (
                    <option key={ft.value} value={ft.value}>
                      {ft.label}
                    </option>
                  ))}
                </select>
                <button
                  className="dsl-remove-btn"
                  data-testid={`btnRemoveFactor-${name}`}
                  onClick={() => removeFactor(name)}
                  disabled={mode === 'free_edit'}
                  title="Remove"
                >
                  ✕
                </button>
              </div>
            ))}
            {factorEntries.length === 0 && (
              <div className="dsl-empty">No factors defined</div>
            )}
          </section>

          {/* Signals Section */}
          <section className="config-section">
            <h3>
              Signals
              <button
                className="dsl-add-btn"
                data-testid="btnAddSignal"
                onClick={addSignal}
                disabled={mode === 'free_edit'}
                title="Add Signal"
              >
                + Add
              </button>
            </h3>
            {signalEntries.map(([name, signal]) => (
              <div className="form-group form-group-row" key={name}>
                <input
                  type="text"
                  value={name}
                  className="form-name-input"
                  readOnly
                  title={`Signal name: ${name}`}
                />
                <select
                  value={signal.type}
                  onChange={(e) => updateSignal(name, e.target.value)}
                  disabled={mode === 'free_edit'}
                >
                  {SIGNAL_TYPES.map((st) => (
                    <option key={st.value} value={st.value}>
                      {st.label}
                    </option>
                  ))}
                </select>
                <button
                  className="dsl-remove-btn"
                  data-testid={`btnRemoveSignal-${name}`}
                  onClick={() => removeSignal(name)}
                  disabled={mode === 'free_edit'}
                  title="Remove"
                >
                  ✕
                </button>
              </div>
            ))}
            {signalEntries.length === 0 && (
              <div className="dsl-empty">No signals defined</div>
            )}
          </section>

          {/* Portfolio Section */}
          <section className="config-section">
            <h3>Portfolio Structure</h3>
            <div className="form-group">
              <label>Portfolio Type</label>
              <select
                value={config.portfolio?.type || 'long_short'}
                onChange={(e) => updatePortfolioType(e.target.value as 'long_short' | 'long_only')}
                disabled={mode === 'free_edit'}
              >
                <option value="long_short">Long/Short</option>
                <option value="long_only">Long Only</option>
              </select>
            </div>
            {config.portfolio?.type === 'long_short' && (
              <>
                <div className="form-group">
                  <label>Long Book Selector</label>
                  <select
                    value={config.portfolio?.long_book?.selector || 'TopN'}
                    disabled={mode === 'free_edit'}
                  >
                    <option value="TopN">Top N</option>
                    <option value="QuantileTop">Quantile Top</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Short Book Selector</label>
                  <select
                    value={config.portfolio?.short_book?.selector || 'BottomN'}
                    disabled={mode === 'free_edit'}
                  >
                    <option value="BottomN">Bottom N</option>
                    <option value="QuantileBottom">Quantile Bottom</option>
                  </select>
                </div>
              </>
            )}
          </section>
        </div>

        {/* Right Panel: Code Editor */}
        <div className="dsl-output">
          {/* Toolbar */}
          <div className="dsl-editor-toolbar" data-testid="dsl-editor-toolbar">
            <div className="dsl-toolbar-left">
              <select
                className="dsl-strategy-select"
                data-testid="dslStrategySelect"
                value={isNewStrategy ? '' : selectedStrategyId}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === '') {
                    setIsNewStrategy(true);
                    setSelectedStrategyId('');
                    setMode('generated');
                  } else {
                    loadStrategy(v);
                  }
                }}
              >
                <option value="">(new strategy)</option>
                {strategies.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id}
                  </option>
                ))}
              </select>

              {isNewStrategy && (
                <input
                  type="text"
                  className="dsl-strategy-name-input"
                  data-testid="dslNewStrategyName"
                  value={newStrategyId}
                  onChange={(e) => setNewStrategyId(e.target.value)}
                  placeholder="Strategy name"
                  title="Name for the new strategy file"
                />
              )}

              <button
                className={`dsl-mode-toggle ${mode === 'free_edit' ? 'dsl-mode-toggle--active' : ''}`}
                data-testid="btnDslModeToggle"
                onClick={toggleMode}
                title={mode === 'generated' ? 'Switch to Free-edit mode' : 'Switch to Generated mode'}
              >
                {mode === 'generated' ? '📝 Generated' : '✏️ Free-edit'}
              </button>
            </div>

            <div className="dsl-toolbar-right">
              <button
                className="btn dsl-action-btn"
                data-testid="btnDslSave"
                onClick={handleSave}
                disabled={isBusy || !editorCode.trim()}
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
              <button
                className="btn dsl-action-btn"
                data-testid="btnDslRun"
                onClick={handleRun}
                disabled={isBusy || !activeStrategyId}
              >
                {running ? 'Running…' : 'Run'}
              </button>
              <button
                className="btn dsl-action-btn dsl-action-btn--primary"
                data-testid="btnDslSaveAndRun"
                onClick={handleSaveAndRun}
                disabled={isBusy || !editorCode.trim()}
              >
                Save &amp; Run
              </button>
            </div>
          </div>

          {/* Toast */}
          {toast && (
            <div
              className={`dsl-toast ${toast.type === 'error' ? 'dsl-toast--error' : 'dsl-toast--success'}`}
              data-testid="dslToast"
            >
              {toast.message}
            </div>
          )}

          {loading && mode === 'generated' && (
            <div className="loading" data-testid="dslLoading">Generating code...</div>
          )}
          {error && <div className="error" data-testid="dslError">Error: {error}</div>}

          {/* Monaco Editor */}
          <div className="dsl-editor-wrapper" data-testid="dsl-editor-wrapper">
            <Editor
              height="100%"
              language="python"
              theme="vs-dark"
              value={editorCode}
              onChange={(v) => setEditorCode(v || '')}
              options={{
                minimap: { enabled: false },
                wordWrap: 'on',
                fontSize: 13,
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                readOnly: false,
                automaticLayout: true,
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
