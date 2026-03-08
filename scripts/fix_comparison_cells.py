"""Fix cells 34 and 35 column name references."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "triple_leveraged_etf_strategy.ipynb"

def lines(code: str) -> list[str]:
    raw = code.strip().split("\n")
    return [l + "\n" for l in raw[:-1]] + [raw[-1]]

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 34 (0-indexed = 33)
CELL_34 = lines("""\
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
    subplot_titles=('Equity: Manual vs DSL (full period)', 'COVID Crash Zoom: Mar-Apr 2020'))

fig.add_trace(go.Scatter(x=combined.index, y=combined['Manual'],
    name='Manual', line=dict(color='blue', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['DSL'],
    name='DSL', line=dict(color='purple', width=2, dash='dot')), row=1, col=1)
fig.update_yaxes(type='log', title_text='Equity ($, log)', row=1, col=1)

covid = combined.loc['2020-02-15':'2020-06-01']
fig.add_trace(go.Scatter(x=covid.index, y=covid['Manual'],
    name='Manual (COVID)', line=dict(color='blue', width=2), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=covid.index, y=covid['DSL'],
    name='DSL (COVID)', line=dict(color='purple', width=2, dash='dot'), showlegend=False), row=2, col=1)

cash_period = strategy_results[strategy_results['in_cash']].index
covid_cash = cash_period[(cash_period >= '2020-02-15') & (cash_period <= '2020-06-01')]
if len(covid_cash) > 0:
    fig.add_vrect(x0=covid_cash[0], x1=covid_cash[-1], fillcolor='orange', opacity=0.15,
                  line_width=0, annotation_text='Manual in cash', row=2, col=1)

fig.update_yaxes(title_text='Equity ($)', row=2, col=1)
fig.update_layout(height=600, template='plotly_white', hovermode='x unified',
                  title='Equity Comparison: Full Period + COVID Crash Detail')
fig.show()""")

nb['cells'][33]['source'] = CELL_34
print(f"Cell 34: {len(CELL_34)} lines")

# Fix cell 35 (0-indexed = 34): replace old column names
cell35 = nb['cells'][34]
src = ''.join(cell35['source'])
src = src.replace("combined['man_equity']", "combined['Manual']")
src = src.replace("combined['dsl_equity']", "combined['DSL']")
raw = src.split('\n')
cell35['source'] = [l + '\n' for l in raw[:-1]] + [raw[-1]]
print(f"Cell 35: {len(cell35['source'])} lines")

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Done")
