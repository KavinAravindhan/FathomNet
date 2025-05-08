# fgvc-comp-2025/model.py
import torch, torch.nn as nn, timm

class HierConvNeXt(nn.Module):
    def __init__(self,
                 num_fine : int = 79,
                 num_coarse: int = 12,
                 backbone  : str = "convnext_large_in22k"):
        super().__init__()
        self.backbone = timm.create_model(backbone,
                                          pretrained=True,
                                          num_classes=0)   # drop classifier
        in_feats = self.backbone.num_features

        self.head_fine    = nn.Linear(in_feats, num_fine)
        self.head_coarse  = nn.Linear(in_feats, num_coarse)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "fine"   : self.head_fine(feat),
            "coarse" : self.head_coarse(feat)
        }
