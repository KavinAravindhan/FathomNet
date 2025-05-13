# fgvc-comp-2025/train.py
from hier_loss_copy import HierarchicalLoss
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd, pathlib, itertools, numpy as np
from dataset   import FathomNetDataset, build_transforms
from taxonomy  import build_maps
from model_copy     import HierCoAtNet
from timm.data import Mixup
from hier_loss import distance_matrix, expected_distance
from timm.scheduler import CosineLRScheduler
# from timm.utils import ModelEmaV2

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf

    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False  # No stopping
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Should stop
        return False
    
# --------------------------------------------------------------------------- #
#  Settings                                                                   #
# --------------------------------------------------------------------------- #
ROOT  = pathlib.Path("fgvc-comp-2025")
CSV   = ROOT / "data/train/annotations.csv"
IMDIR = ROOT / "data/train"
CKDIR = pathlib.Path("model_checkpoints_copy/focal_loss"); CKDIR.mkdir(exist_ok=True)

PHASES = [               # (input_res, batch, epochs)
    (224, 64, 20),
    # (384, 16, 8),
    # (512,  8, 6),
]
LR_HEAD = 1e-4
LR_BODY = 1e-5

DEVICE = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(DEVICE)
if DEVICE=="cpu":
    exit()
# --------------------------------------------------------------------------- #
#  Data split & label maps                                                    #
# --------------------------------------------------------------------------- #
df        = pd.read_csv(CSV)
train_idx, val_idx = train_test_split(
    df.index, test_size=0.2, stratify=df["label"], random_state=42)
df_train = df.loc[train_idx].reset_index(drop=True)
df_val   = df.loc[val_idx].reset_index(drop=True)

train_csv = ROOT / "data/train_split.csv"; df_train.to_csv(train_csv, index=False)
val_csv   = ROOT / "data/val_split.csv";   df_val.to_csv  (val_csv,   index=False)

name2idx, idx2name, coarse_of_idx, coarse_names = build_maps(CSV)
num_coarse = len(coarse_names)
D = distance_matrix(tuple(idx2name))       # numpy 79×79

# --------------------------------------------------------------------------- #
#  Datasets (transforms resized per phase)                                    #
# --------------------------------------------------------------------------- #
train_ds = FathomNetDataset(train_csv, IMDIR, name2idx, use_roi=True, split="train")
val_ds   = FathomNetDataset(val_csv,   IMDIR, name2idx, use_roi=True, split="val")

# --------------------------------------------------------------------------- #
#  Model, optimizer, MixUp                                                   #
# --------------------------------------------------------------------------- #
model = HierCoAtNet(num_fine=79, num_coarse=num_coarse).to(DEVICE)

head_params = itertools.chain(model.head_fine.parameters(),
                              model.head_coarse.parameters())
# optimizer = optim.AdamW([
#     {"params": model.backbone.parameters(), "lr": LR_BODY},
#     {"params": head_params,                 "lr": LR_HEAD},
# ])

# backbone gets lr × 0.25, heads lr × 1.0
backbone_params = list(model.backbone.parameters())
opt_groups = [
    {"params": backbone_params, "lr": LR_BODY, "weight_decay": 0.2},
    {"params": head_params,     "lr": LR_HEAD, "weight_decay": 0.2},
]
optimizer = optim.AdamW(opt_groups, betas=(0.9,0.999))

# scheduler
sched = CosineLRScheduler(
    optimizer, t_initial=sum(p[2] for p in PHASES),
    lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-6)

# ema = ModelEmaV2(model, decay=0.9999, device=DEVICE)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type != "cpu"))

mixup_fn = Mixup(
    mixup_alpha=1.2,  # Increased from 0.8
    cutmix_alpha=1.2,
    prob=0.8,  # Reduced from 1.0
    mode='batch'
)

# # Gradient Clipping (after backward())
# scaler.unscale_(optimizer)
# torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

# Enhanced Augmentations
train_ds.tfm = build_transforms("train", 224)

# --------------------------------------------------------------------------- #
#  Training                                                                   #
# --------------------------------------------------------------------------- #
epoch_global = 1
best_ckpts = []  # List of tuples: (val_acc, ckpt_path)
global_best_acc = -np.inf
global_best_path = None

for size, batch, n_epochs in PHASES:
    print(f"\n### Phase {size}px  {n_epochs} epochs ###")

    # Phase-specific early stopper (patience=1/3 of phase epochs)
    early_stopper = EarlyStopper(patience=max(4, n_epochs//3), min_delta=0.001)

    train_ds.tfm = build_transforms("train", size)
    val_ds.tfm   = build_transforms("val",   size)

    train_loader = DataLoader(train_ds, batch, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch, shuffle=False,
                              num_workers=2)

    if size > 224:                   # shrink LR for higher-res phases
        for g in optimizer.param_groups:
            g["lr"] *= 0.1
    
     # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    freeze_epochs = 1

    # Track best model in current phase
    phase_best_acc = -np.inf
    phase_best_path = None

    for epoch_idx in range(n_epochs):
        if epoch_idx == freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True

        # Training loop (keep existing code)
        model.train()
        tot_loss = 0
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"phase{size}_e{epoch_global}"):

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            imgs, labels_mix = mixup_fn(imgs, labels)
            coarse_lbl  = torch.tensor([coarse_of_idx[l.item()] for l in labels],
                                       device=DEVICE)
            coarse_mix  = F.one_hot(coarse_lbl, num_classes=num_coarse).float()
            coarse_mix *= labels_mix.sum(dim=1, keepdim=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                out         = model(imgs)
                loss_fine   = (-labels_mix  * F.log_softmax(out["fine"],   1)).sum(1).mean()
                loss_coarse = (-coarse_mix * F.log_softmax(out["coarse"], 1)).sum(1).mean()
                loss_hier   = expected_distance(out["fine"], labels, D)
                # loss = loss_fine + 0.5*loss_coarse + 0.3*loss_hier
                # loss = HierarchicalLoss(coarse_of_idx, D=D)(out["fine"], labels, D)
                # Added by Kavin
                loss = HierarchicalLoss(coarse_of_idx)(out["fine"], labels, D)
            scaler.scale(loss).backward()
            # scaler.step(optimizer); scaler.update()
            scaler.unscale_(optimizer)  # Now safe - scaler initialized
        
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # Optimizer step (scaled weights)
            scaler.step(optimizer)
            
            # Update scaler
            scaler.update()
            tot_loss += loss.item() * imgs.size(0)

        # -------- Validation --------
        model.eval(); hit=0; tot=0; dist_sum=0
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)["fine"]
                pred   = logits.argmax(1)
                hit   += (pred == labels).sum().item()
                tot   += labels.size(0)
                dist_sum += expected_distance(logits, labels, D).item()*imgs.size(0)

        val_acc  = hit / tot
        val_dist = dist_sum / tot
        print(f"Epoch {epoch_global:02d}  "
              f"train-loss {tot_loss/len(train_loader.dataset):.4f}  "
              f"val-acc {val_acc:.3%}  mean-dist {val_dist:.3f}")
        
        # Early stopping check
        if early_stopper(val_acc):
            print(f"Early stopping triggered at epoch {epoch_global} (best phase acc: {early_stopper.best_score:.3%})")
            break

        # Save best phase checkpoint
        if val_acc > phase_best_acc:
            phase_best_acc = val_acc
            if phase_best_path:
                phase_best_path.unlink(missing_ok=True)
            phase_best_path = CKDIR / f"best_phase{size}.pt"
            torch.save(model.state_dict(), phase_best_path)

            # Update global best
            if val_acc > global_best_acc:
                global_best_acc = val_acc
                if global_best_path:
                    global_best_path.unlink(missing_ok=True)
                global_best_path = CKDIR / "best_global.pt"
                torch.save(model.state_dict(), global_best_path)

        # Existing scheduler and EMA update
        sched.step(epoch_global)
        # ema.update(model)
        epoch_global += 1

print("\n✓ Training complete")
print(f"Global best accuracy: {global_best_acc:.3%}")
print(f"Best model saved at: {global_best_path}")


# nohup python train_copy.py > fathomnet_heir_loss_with_focal.log 2>&1 &