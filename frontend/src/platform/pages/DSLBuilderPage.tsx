import React, { useEffect, useState } from 'react';
import './DSLBuilderPage.css';

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

async function fetchDSLCode(config: DSLConfig): Promise<DSLCode> {
  const res = await fetch('/api/dsl/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return (await res.json()) as DSLCode;
}

export function DSLBuilderPage() {
  const [config, setConfig] = useState<DSLConfig>({
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
  });

  const [dslCode, setDSLCode] = useState<DSLCode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    fetchDSLCode(config)
      .then(setDSLCode)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [config]);

  const updateDataConfig = (key: keyof (typeof config)['data'], value: string) => {
    setConfig((prev) => ({
      ...prev,
      data: { ...prev.data!, [key]: value },
    }));
  };

  const updateUniverseName = (name: string) => {
    setConfig((prev) => ({
      ...prev,
      universe: { ...prev.universe!, name },
    }));
  };

  const updateFactor = (name: string, type: string) => {
    setConfig((prev) => ({
      ...prev,
      factors: { ...prev.factors!, [name]: { type, params: { lookback: 126 } } },
    }));
  };

  const updatePortfolioType = (type: 'long_short' | 'long_only') => {
    setConfig((prev) => ({
      ...prev,
      portfolio: { ...prev.portfolio!, type },
    }));
  };

  return (
    <div className="dsl-builder">
      <div className="dsl-container">
        {/* Left Panel: Controls */}
        <div className="dsl-controls">
          <h2>DSL Strategy Builder</h2>

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
              />
            </div>
            <div className="form-group">
              <label>Calendar</label>
              <select
                value={config.data?.calendar || 'XNYS'}
                onChange={(e) => updateDataConfig('calendar', e.target.value)}
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
              />
            </div>
            <div className="form-group">
              <label>End Date</label>
              <input
                type="date"
                value={config.data?.end_date || '2025-12-31'}
                onChange={(e) => updateDataConfig('end_date', e.target.value)}
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
              />
            </div>
          </section>

          {/* Factors Section */}
          <section className="config-section">
            <h3>Factors</h3>
            <div className="form-group">
              <label>Momentum Factor</label>
              <select
                value={config.factors?.mom_126?.type || 'momentum'}
                onChange={(e) => updateFactor('mom_126', e.target.value)}
              >
                <option value="momentum">Momentum (126-day)</option>
                <option value="volatility">Volatility (63-day)</option>
                <option value="mean_reversion">Mean Reversion</option>
              </select>
            </div>
          </section>

          {/* Portfolio Section */}
          <section className="config-section">
            <h3>Portfolio Structure</h3>
            <div className="form-group">
              <label>Portfolio Type</label>
              <select
                value={config.portfolio?.type || 'long_short'}
                onChange={(e) => updatePortfolioType(e.target.value as 'long_short' | 'long_only')}
              >
                <option value="long_short">Long/Short</option>
                <option value="long_only">Long Only</option>
              </select>
            </div>
            {config.portfolio?.type === 'long_short' && (
              <>
                <div className="form-group">
                  <label>Long Book Selector</label>
                  <select value={config.portfolio?.long_book?.selector || 'TopN'}>
                    <option value="TopN">Top N</option>
                    <option value="QuantileTop">Quantile Top</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Short Book Selector</label>
                  <select value={config.portfolio?.short_book?.selector || 'BottomN'}>
                    <option value="BottomN">Bottom N</option>
                    <option value="QuantileBottom">Quantile Bottom</option>
                  </select>
                </div>
              </>
            )}
          </section>
        </div>

        {/* Right Panel: DSL Code Output */}
        <div className="dsl-output">
          <h2>Generated DSL Code</h2>
          {loading && <div className="loading">Generating code...</div>}
          {error && <div className="error">Error: {error}</div>}
          {dslCode && (
            <>
              <div className="code-tabs">
                <button className="tab-btn active">Python Code</button>
                <button className="tab-btn">JSON Config</button>
              </div>
              <pre className="code-block">
                <code>{dslCode.python_code}</code>
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
