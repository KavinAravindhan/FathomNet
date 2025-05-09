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

# ---------- ADD just below HierConvNeXt class ----------
class HierSwinB(HierConvNeXt):
    """Swin-V2 Base, ImageNet-22k pretrained"""
    def __init__(self, num_fine=79, num_coarse=12):
        super().__init__(num_fine, num_coarse,
            backbone="swinv2_base_window12to24_192to384.ms_in22k")

class HierViTMAE(HierConvNeXt):
    """ViT-Base MAE-finetuned checkpoint"""
    def __init__(self, num_fine=79, num_coarse=12):
        super().__init__(num_fine, num_coarse,
            backbone="vit_base_patch16_384.mae")

