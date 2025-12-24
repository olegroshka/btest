import { escapeHtml } from './api.js';

export function mountWorkspace(hostId = 'workspace') {
  const host = document.getElementById(hostId);
  if (!host) return;

  host.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <div style="font-weight:650">Workspace</div>
      <div id="canvasTabs" style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="tab" data-tab="plot" id="tabPlot">Plot</button>
        <button class="tab" data-tab="table" id="tabTable">Table</button>
        <button class="tab" data-tab="raw" id="tabRaw">Raw</button>
      </div>
    </div>

    <div id="plotWrap" style="margin-top:10px">
      <div style="color:var(--muted);font-size:12px;margin-bottom:6px"><b>Price chart</b> <span id="plotStatus">(idle)</span></div>
      <div id="plot" data-testid="plotly-chart" class="card" style="height:360px;padding:6px;overflow:hidden">(no data)</div>
    </div>

    <div id="tableWrap" style="display:none;margin-top:10px">
      <div id="previewSummary" style="margin-top:6px"></div>
      <div id="previewTables" style="margin-top:10px"></div>
    </div>

    <div id="rawWrap" style="display:none;margin-top:10px">
      <pre id="previewRaw" style="overflow:auto;max-height:520px"></pre>
    </div>
  `;

  function setTab(name) {
    const isPlot = name === 'plot';
    const isTable = name === 'table';
    const isRaw = name === 'raw';

    const pw = document.getElementById('plotWrap');
    const tw = document.getElementById('tableWrap');
    const rw = document.getElementById('rawWrap');
    if (pw) pw.style.display = isPlot ? 'block' : 'none';
    if (tw) tw.style.display = isTable ? 'block' : 'none';
    if (rw) rw.style.display = isRaw ? 'block' : 'none';

    const bPlot = document.getElementById('tabPlot');
    const bTable = document.getElementById('tabTable');
    const bRaw = document.getElementById('tabRaw');
    if (bPlot) bPlot.disabled = isPlot;
    if (bTable) bTable.disabled = isTable;
    if (bRaw) bRaw.disabled = isRaw;
  }

  // initial
  setTab('plot');

  const tabs = host.querySelectorAll('#canvasTabs [data-tab]');
  for (const t of tabs) {
    t.addEventListener('click', (ev) => {
      ev.preventDefault();
      const name = t.getAttribute('data-tab');
      if (name) setTab(name);
    });
  }

  return { setTab };
}

export function renderPreviewTables(data) {
  const summary = document.getElementById('previewSummary');
  const tables = document.getElementById('previewTables');
  const raw = document.getElementById('previewRaw');

  if (!tables || !summary || !raw) return;

  const rows = (data && typeof data.rows === 'number') ? data.rows : '';
  const idx0 = data ? (data.index_start || '') : '';
  const idx1 = data ? (data.index_end || '') : '';

  summary.innerHTML = `<span style="display:inline-block;margin-right:8px">rows: <b>${escapeHtml(String(rows))}</b></span>` +
    (idx0 ? ` <span style="margin-right:8px">start: <code>${escapeHtml(String(idx0))}</code></span>` : '') +
    (idx1 ? ` <span style="margin-right:8px">end: <code>${escapeHtml(String(idx1))}</code></span>` : '');

  function renderTable(title, arr) {
    if (!Array.isArray(arr) || !arr.length) return `<div style="color:#666">(${escapeHtml(title)} empty)</div>`;
    const cols = Array.isArray(data.columns) && data.columns.length ? data.columns : Object.keys(arr[0] || {});
    let h = `<div style="color:#666;margin:6px 0"><b>${escapeHtml(title)}</b> (${arr.length})</div>`;
    h += '<div style="overflow:auto;border:1px solid #eee;border-radius:10px">';
    h += '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    h += '<thead><tr style="text-align:left;background:#fafafa">';
    for (const c of cols) h += `<th style="padding:8px 10px;border-bottom:1px solid #eee">${escapeHtml(c)}</th>`;
    h += '</tr></thead><tbody>';
    for (const r of arr) {
      h += '<tr>';
      for (const c of cols) {
        const v = (r && Object.prototype.hasOwnProperty.call(r, c)) ? r[c] : '';
        h += `<td style="padding:8px 10px;border-bottom:1px solid #f3f3f3">${escapeHtml(v ?? '')}</td>`;
      }
      h += '</tr>';
    }
    h += '</tbody></table></div>';
    return h;
  }

  tables.innerHTML = renderTable('head', data.head) + '<div style="height:10px"></div>' + renderTable('tail', data.tail);
  raw.textContent = JSON.stringify(data, null, 2);
}

export function renderPlotlyFromChartPayload(payload) {
  const el = document.getElementById('plot');
  const st = document.getElementById('plotStatus');
  if (!el) return;

  const Plotly = window.Plotly;
  if (!Plotly) {
    el.textContent = '(plotly not available)';
    if (st) st.textContent = '(no plotly)';
    return;
  }

  try {
    const cols = Array.isArray(payload?.columns) ? payload.columns : [];
    const data = Array.isArray(payload?.data) ? payload.data : [];
    if (!data.length) {
      el.textContent = '(no data)';
      if (st) st.textContent = '(no data)';
      return;
    }

    const ts = data.map((r) => r.ts);

    const lowerCols = cols.map((c) => String(c).toLowerCase());
    const idxOpen = lowerCols.indexOf('open');
    const idxHigh = lowerCols.indexOf('high');
    const idxLow = lowerCols.indexOf('low');
    const idxClose = lowerCols.indexOf('close');
    const idxVolume = lowerCols.indexOf('volume');

    const hasOhlc = idxOpen >= 0 && idxHigh >= 0 && idxLow >= 0 && idxClose >= 0;
    const hasVol = idxVolume >= 0;

    let traces = [];
    if (hasOhlc) {
      const open = data.map((r) => r[cols[idxOpen]]);
      const high = data.map((r) => r[cols[idxHigh]]);
      const low = data.map((r) => r[cols[idxLow]]);
      const close = data.map((r) => r[cols[idxClose]]);

      // Stronger colors for dark theme
      const up = '#14f195';
      const down = '#ff3b6b';

      // Candlestick trace
      traces = [{
        type: 'candlestick',
        x: ts,
        open, high, low, close,
        name: 'OHLC',
        increasing: { line: { color: up, width: 1.2 } },
        decreasing: { line: { color: down, width: 1.2 } },
        hoverlabel: { align: 'left' },
        hovertemplate:
          'Time: %{x|%Y-%m-%d}<br>' +
          'Open: %{open:.2f}<br>' +
          'High: %{high:.2f}<br>' +
          'Low: %{low:.2f}<br>' +
          'Close: %{close:.2f}<br>' +
          '<extra></extra>',
      }];

      // Volume bars (secondary axis) if available
      if (hasVol) {
        const vol = data.map((r) => r[cols[idxVolume]]);
        const volColors = close.map((c, i) => (c >= open[i] ? 'rgba(20,241,149,0.35)' : 'rgba(255,59,107,0.35)'));
        traces.push({
          type: 'bar',
          x: ts,
          y: vol,
          name: 'Volume',
          yaxis: 'y2',
          marker: { color: volColors },
          hovertemplate:
            'Time: %{x|%Y-%m-%d}<br>' +
            'Volume: %{y:,}<br>' +
            '<extra></extra>',
        });
      }
    } else {
      const c0 = cols.find((c) => c && String(c).toLowerCase() !== 'ts');
      if (!c0) {
        el.textContent = '(no plottable columns)';
        if (st) st.textContent = '(no plottable columns)';
        return;
      }
      const y = data.map((r) => r[c0]);
      traces = [{
        type: 'scatter',
        mode: 'lines',
        x: ts,
        y,
        name: String(c0),
        line: { color: '#22d3ee', width: 1.8 },
        hovertemplate: 'Time: %{x|%Y-%m-%d}<br>' + escapeHtml(String(c0)) + ': %{y:.4g}<br><extra></extra>',
      }];

      if (hasVol) {
        const vol = data.map((r) => r[cols[idxVolume]]);
        traces.push({
          type: 'bar',
          x: ts,
          y: vol,
          name: 'Volume',
          yaxis: 'y2',
          marker: { color: 'rgba(34,211,238,0.20)' },
          hovertemplate:
            'Time: %{x|%Y-%m-%d}<br>' +
            'Volume: %{y:,}<br>' +
            '<extra></extra>',
        });
      }
    }

    const rootStyle = getComputedStyle(document.documentElement);
    const bg = rootStyle.getPropertyValue('--bg').trim() || '#0b1020';
    const panel = rootStyle.getPropertyValue('--panel').trim() || '#0f172a';
    const border = rootStyle.getPropertyValue('--border').trim() || '#24304a';
    const text = rootStyle.getPropertyValue('--text').trim() || '#e5e7eb';
    const muted = rootStyle.getPropertyValue('--muted').trim() || '#9aa4b2';

    const layout = {
      margin: { l: 55, r: 20, t: 10, b: 40 },
      paper_bgcolor: panel,
      plot_bgcolor: panel,
      font: { color: text },
      xaxis: {
        type: 'date',
        fixedrange: false,
        gridcolor: border,
        tickfont: { color: muted },
        // Make the range slider mini-chart readable but not dominant.
        rangeslider: { visible: true, bgcolor: bg, bordercolor: border, thickness: 0.14 },
        rangeselector: {
          bgcolor: panel,
          activecolor: '#7c3aed',
          bordercolor: border,
          borderwidth: 1,
          font: { color: text, size: 11 },
          buttons: [
            { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
            { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
            { count: 1, label: 'YTD', step: 'year', stepmode: 'todate' },
            { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
            { count: 5, label: '5y', step: 'year', stepmode: 'backward' },
            { step: 'all', label: 'All' },
          ],
        },
      },
      yaxis: {
        fixedrange: false,
        gridcolor: border,
        tickfont: { color: muted },
        // Give volume meaningful height.
        domain: hasVol ? [0.34, 1] : [0, 1],
        title: '',
      },
      yaxis2: hasVol ? {
        fixedrange: false,
        gridcolor: border,
        tickfont: { color: muted },
        domain: [0, 0.30],
        title: 'Vol',
        showticklabels: false,
        zeroline: false,
      } : undefined,
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor: panel,
        bordercolor: border,
        font: { color: text },
      },
      showlegend: false,
      bargap: 0.05,
    };

    Plotly.react(
      el,
      traces,
      layout,
      {
        displayModeBar: true,
        responsive: true,
        scrollZoom: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
      }
    );

    // Ensure plot is visible after re-render (tab switches)
    try { Plotly.Plots.resize(el); } catch (e) {}

    if (st) st.textContent = '(ready)';
  } catch (e) {
    el.textContent = '(plot error)';
    if (st) st.textContent = '(plot error)';
  }
}
