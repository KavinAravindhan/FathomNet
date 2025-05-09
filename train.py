# fgvc-comp-2025/train.py
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd, pathlib, itertools, numpy as np
from dataset   import FathomNetDataset, build_transforms
from taxonomy  import build_maps
from model     import HierConvNeXt
from timm.data import Mixup
from hier_loss import distance_matrix, expected_distance

# --------------------------------------------------------------------------- #
#  Settings                                                                   #
# --------------------------------------------------------------------------- #
ROOT  = pathlib.Path("fgvc-comp-2025")
CSV   = ROOT / "data/train/annotations.csv"
IMDIR = ROOT / "data/train"
CKDIR = pathlib.Path("model_checkpoints/using_worm"); CKDIR.mkdir(exist_ok=True)

PHASES = [               # (input_res, batch, epochs)
    (224, 32, 10),
    (384, 16, 8),
    (512,  8, 6),
]
LR_HEAD = 1e-4
LR_BODY = 1e-5

DEVICE = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
model = HierConvNeXt(num_fine=79, num_coarse=num_coarse).to(DEVICE)

head_params = itertools.chain(model.head_fine.parameters(),
                              model.head_coarse.parameters())
optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": LR_BODY},
    {"params": head_params,                 "lr": LR_HEAD},
])
scaler = torch.amp.GradScaler(enabled=(DEVICE.type != "cpu"))

mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0,
    prob=1.0, switch_prob=0.5, mode="elem",
    label_smoothing=0.1, num_classes=79
)

# --------------------------------------------------------------------------- #
#  Training                                                                   #
# --------------------------------------------------------------------------- #
epoch_global = 1
for size, batch, n_epochs in PHASES:
    print(f"\n### Phase {size}px  {n_epochs} epochs ###")
    train_ds.tfm = build_transforms("train", size)
    val_ds.tfm   = build_transforms("val",   size)

    train_loader = DataLoader(train_ds, batch, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch, shuffle=False,
                              num_workers=2)

    if size > 224:                   # shrink LR for higher-res phases
        for g in optimizer.param_groups:
            g["lr"] *= 0.1

    for _ in range(n_epochs):
        model.train(); tot_loss = 0
        for imgs, labels in tqdm(train_loader, leave=False):
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
                loss = loss_fine + 0.5*loss_coarse + 0.3*loss_hier
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
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

        torch.save(model.state_dict(),
                   CKDIR / f"ckpt_s{size}_e{epoch_global}.pt")
        epoch_global += 1

print("\n✓ Training complete – checkpoints stored in", CKDIR)


# nohup python train.py > fathomnet_worm.log 2>&1 &