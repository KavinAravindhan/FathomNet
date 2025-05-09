# fgvc-comp-2025/hier_loss.py
"""
Exact WoRMS distance matrix + expected-distance loss.

If 'fathomnet_taxonomy.json' is missing, we fetch all 79 labels from the
FathomNet WoRMS API on the fly and save the JSON locally for next runs.
"""
import json, numpy as np, torch, pathlib, requests, time
from functools import lru_cache

JSON_PATH = pathlib.Path(__file__).with_name("fathomnet_taxonomy.json")
BASE_URL  = "https://fathomnet.org/api/worms/{}"          # slug or aphia id

# --------------------------------------------------------------------------- #
#  Fetch and cache taxonomy once                                              #
# --------------------------------------------------------------------------- #
def _download_tree(class_names):
    print("↯  building taxonomy JSON from remote API (one-time)…")
    tree = {}
    for name in class_names:
        slug = name.replace(" ", "%20")
        for _ in range(3):                                 # retry up to 3×
            r = requests.get(BASE_URL.format(slug), timeout=10)
            if r.status_code == 200:
                break
            time.sleep(2)
        data = r.json()
        # Walk up parent chain
        cur = data
        while cur:
            tree[cur["scientificName"]] = {
                "parent": cur.get("parentNameUsage")
            }
            cur = cur.get("parent")
    JSON_PATH.write_text(json.dumps(tree))
    print(f"✓  saved taxonomy JSON → {JSON_PATH}")

def _ensure_json(names):
    if not JSON_PATH.exists():
        _download_tree(names)

# --------------------------------------------------------------------------- #
#  Utilities                                                                  #
# --------------------------------------------------------------------------- #
def _ancestors(name, tree):
    anc = []
    cur = name
    while cur and cur in tree:
        anc.append(cur)
        cur = tree[cur]["parent"]
    return anc

@lru_cache(maxsize=1)
def distance_matrix(tuple_names):
    names = list(tuple_names)
    _ensure_json(names)
    tree  = json.loads(JSON_PATH.read_text())
    anc   = {n: _ancestors(n, tree) for n in names}

    n = len(names)
    D = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j: continue
            la, lb = anc[a], anc[b]
            lca = 0
            for u, v in zip(reversed(la), reversed(lb)):
                if u == v: lca += 1
                else: break
            D[i, j] = len(la) + len(lb) - 2 * lca
    return D

def expected_distance(logits, target, D):
    P   = torch.softmax(logits, 1)                        # [B,79]
    Dt  = torch.from_numpy(D[target.cpu()]).to(logits)    # [B,79]
    return (P * Dt).sum(1).mean()
