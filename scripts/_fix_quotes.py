"""Fix escaped triple quotes in notebook cells."""
import json

NB = "notebooks/explore_cac_price_vs_vol.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixed = 0
for cell in nb["cells"]:
    new_src = []
    changed = False
    for line in cell.get("source", []):
        if '\\"' in line:
            line = line.replace('\\"', '"')
            changed = True
        new_src.append(line)
    if changed:
        cell["source"] = new_src
        fixed += 1

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"Fixed escaped quotes in {fixed} cells")
