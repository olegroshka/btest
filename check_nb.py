import json

with open('notebooks/explore_indices_signals.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("Notebook metadata:", nb.get('metadata', {}))
print()

for i, c in enumerate(nb['cells'][:10]):
    meta = c.get('metadata', {})
    print(f"Cell {i} type={c['cell_type']}: metadata={meta}")
    if 'outputs' in c:
        print(f"  Has {len(c['outputs'])} outputs")
