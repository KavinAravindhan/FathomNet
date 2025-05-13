import json, numpy as np, torch, pathlib
from functools import lru_cache
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalLoss(nn.Module):
    def __init__(self, coarse_of_idx, gamma=2, depth_weights=[0.1, 0.3, 0.5, 0.7]):
        super().__init__()
        self.coarse_of_idx = coarse_of_idx
        self.gamma = gamma
        self.depth_weights = torch.tensor(depth_weights)
        
    def forward(self, logits, targets, D):
        # Base loss components
        loss_fine = F.nll_loss(F.log_softmax(logits, 1), targets)
        
        # PROPER TENSOR CONVERSION
        coarse_targets = torch.tensor(
            [self.coarse_of_idx[t.item()] for t in targets],
            device=targets.device,
            dtype=torch.long  # Explicit dtype for integer labels
        )
        
        # Hierarchical components
        loss_hier = self.focal_expected_distance(logits, targets, D)
        loss_consistency = self.hierarchical_consistency(logits, coarse_targets)  # Use tensor here
        
        # Adaptive weighting
        h_loss_weight = nn.Parameter(torch.tensor(0.3))
        consistency_weight = nn.Parameter(torch.tensor(0.2))
        
        total_loss = loss_fine + h_loss_weight * loss_hier + consistency_weight * loss_consistency
        return total_loss

    
    def focal_expected_distance(self, logits, targets, D):
        P = torch.softmax(logits, 1)
        Dt = torch.from_numpy(D[targets.cpu()]).to(logits)
        focal_weights = (1 - Dt) ** self.gamma
        return (P * Dt * focal_weights).sum(1).mean()
    
    # def hierarchical_consistency(self, logits, coarse_labels):
    #     predicted_coarse = torch.tensor([self.coarse_of_idx[l.item()] for l in logits.argmax(1)], device=logits.device)
    #     consistency_mask = (predicted_coarse == coarse_labels).float()
    #     return (1 - consistency_mask).mean()
    
    # Added by Kavin
    def hierarchical_consistency(self, logits, coarse_labels):
        # Vectorized coarse label mapping
        fine_preds = logits.argmax(1)
        coarse_mapping = torch.tensor(
            [self.coarse_of_idx[i] for i in range(logits.size(1))],
            device=logits.device
        )
        predicted_coarse = coarse_mapping[fine_preds]
        
        # Ensure comparison between tensors
        consistency_mask = (predicted_coarse == coarse_labels).float()
        return (1 - consistency_mask).mean()


