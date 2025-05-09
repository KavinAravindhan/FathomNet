# fgvc-comp-2025/hier_loss.py
"""
Offline hierarchical-distance utilities
---------------------------------------
We avoid every network/API call by building an *approximate*
distance matrix from the labels themselves:

• distance = 0  → identical class                     (species / genus / family…)
• distance = 1  → different class **but same first token** (usually family)
• distance = 6  → everything else (worst-case score in the Kaggle metric)

This heuristic matches what many top Kaggle baselines do and
requires **no external taxonomy** – it always works.

If you later obtain the organisers’ `taxonomy_map.json` you can
drop it next to this file and the code will auto-upgrade to the
exact competition tree without changing anything else.
"""
import json, numpy as np, torch, pathlib
from functools import lru_cache

_MAP = pathlib.Path(__file__).with_name("taxonomy_map.json")  # optional

# ------------------------------------------------------------------ #
# 1) build official matrix if JSON exists                            #
# ------------------------------------------------------------------ #
def _matrix_from_json(names):
    tax = json.loads(_MAP.read_text())
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    paths = {n: [tax[n][r] for r in ranks if tax[n][r]] for n in names}
    n = len(names)
    D = np.zeros((n, n), np.float32)
    for i, a in enumerate(names):
        pa = paths[a]
        for j, b in enumerate(names):
            if i == j: continue
            pb = paths[b]
            lca = sum(u == v for u, v in zip(pa, pb))
            D[i, j] = len(pa) + len(pb) - 2 * lca
    return D

# ------------------------------------------------------------------ #
# 2) very-robust fallback                                            #
# ------------------------------------------------------------------ #
def _matrix_fast(names):
    n = len(names)
    first = [s.split()[0] for s in names]               # token 0 ≈ family
    D = np.full((n, n), 6, np.float32)
    for i in range(n):
        D[i, i] = 0
        for j in range(i+1, n):
            if first[i] == first[j]:
                D[i, j] = D[j, i] = 1
    return D

@lru_cache(maxsize=1)
def distance_matrix(tuple_names):
    names = list(tuple_names)
    if _MAP.exists():
        try:
            return _matrix_from_json(names)
        except Exception as e:
            print("⚠ taxonomy_map.json unreadable – falling back:", e)
    return _matrix_fast(names)

def expected_distance(logits, target, D):
    P  = torch.softmax(logits, 1)                       # [B,79]
    Dt = torch.from_numpy(D[target.cpu()]).to(logits)   # [B,79]
    return (P * Dt).sum(1).mean()
