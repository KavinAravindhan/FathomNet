import json, numpy as np, torch, pathlib
from functools import lru_cache
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalLoss(nn.Module):
    def __init__(self, coarse_of_idx, gamma=2, num_classes=79):
        super().__init__()
        self.coarse_of_idx = coarse_of_idx
        self.gamma = gamma
        
        # Learnable weights
        self.register_parameter('h_weight', nn.Parameter(torch.tensor(0.3)))
        self.register_parameter('c_weight', nn.Parameter(torch.tensor(0.2)))
        
        # Precomputed mapping (device-safe)
        self.register_buffer('coarse_mapping', 
            torch.tensor([coarse_of_idx[i] for i in range(num_classes)], dtype=torch.long)
        )

    def forward(self, logits, targets, D):
        # Label smoothing
        loss_fine = F.cross_entropy(logits, targets, label_smoothing=0.2)
        
        # Device-safe targets
        coarse_targets = self.coarse_mapping[targets]
        
        # Hierarchical components
        loss_hier = self.focal_distance(logits, targets, D)
        loss_consistency = (self.coarse_mapping[logits.argmax(1)] != coarse_targets).float().mean()
        
        return loss_fine + self.h_weight*loss_hier + self.c_weight*loss_consistency

    def focal_distance(self, logits, targets, D):
        P = torch.softmax(logits, 1)
        Dt = torch.from_numpy(D[targets.cpu().numpy()]).to(logits.device).float()
        
        # Stability fixes
        Dt = (Dt - Dt.min()) / (Dt.max() - Dt.min() + 1e-8)
        Dt = torch.clamp(Dt, min=1e-4, max=1.0)
        
        return (P * Dt * (1 - Dt)**self.gamma).sum(1).mean()
