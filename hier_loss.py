import numpy as np
from functools import lru_cache
import torch, torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Build a 79×79 taxonomic distance matrix once.                               #
# --------------------------------------------------------------------------- #
def distance_matrix(class_names):
    """
    Very simple rule: distance = 0 if same, 1 if share first token (family),
    else 6.  Replace with full WoRMS lookup later.
    """
    n = len(class_names)
    D = np.full((n, n), 6, dtype=np.float32)
    for i, a in enumerate(class_names):
        for j, b in enumerate(class_names):
            if i == j:
                D[i, j] = 0
            elif a.split()[0] == b.split()[0]:
                D[i, j] = 1
    return D

@lru_cache(maxsize=1)
def cached_D(tupled_names):
    return distance_matrix(list(tupled_names))

def expected_distance(logits, target, D):
    """
    logits : Tensor [B, 79]  (pre-softmax)
    target : Tensor [B]      (int labels)
    D      : numpy [79, 79]  distance matrix
    Returns:
        Tensor scalar = mean expected tree distance for the batch
    """
    P = F.softmax(logits, dim=1)               # [B,79]
    dist_target = torch.from_numpy(D[target.cpu()])\
                      .to(logits.device)       # [B,79]
    exp_dist = (P * dist_target).sum(dim=1)    # [B]
    return exp_dist.mean()
