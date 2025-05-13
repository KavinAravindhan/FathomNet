# fgvc-comp-2025/train_all.py
"""
Train four back-bones sequentially (no EMA, no progressive resizing).

Usage
-----
python train_all.py            # trains effv2m, convnext, swin, vit
python train_all.py --arch effv2m swin   # train subset
"""
import argparse, pathlib, pandas as pd, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import FathomNetDataset, build_transforms
from taxonomy import build_maps
from hier_loss import distance_matrix, expected_distance
from losses import FocalLoss
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler
import torch.optim as optim
from model import HierConvNeXt, HierSwinB, HierViTMAE, HierEffV2M

# -------------------------------------------------------------------- #
#  CLI                                                                  #
# -------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--arch", nargs="+",
                    default=["effv2m","convnext","swin","vit"],
                    choices=["effv2m","convnext","swin","vit"])
args = parser.parse_args()

ARCH2CLS = {
    "effv2m":  HierEffV2M,
    "convnext":HierConvNeXt,
    "swin":    HierSwinB,
    "vit":     HierViTMAE,
}

# -------------------------------------------------------------------- #
#  CONSTANTS                                                           #
# -------------------------------------------------------------------- #
ROOT   = pathlib.Path("fgvc-comp-2025")
CSV    = ROOT/"data/train/annotations.csv"
IMDIR  = ROOT/"data/train"
CKDIR  = pathlib.Path("model_checkpoints/train_all"); CKDIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 15                         # at 384 px
BATCH  = 16
LR     = 3e-4

# -------------------------------------------------------------------- #
#  DATA SPLIT                                                          #
# -------------------------------------------------------------------- #
df = pd.read_csv(CSV)
train_idx, val_idx = train_test_split(df.index, stratify=df["label"],
                                      test_size=0.2, random_state=42)
df.loc[train_idx].to_csv(ROOT/"train_split.csv", index=False)
df.loc[val_idx].to_csv  (ROOT/"val_split.csv",   index=False)

name2idx, idx2name, _, _ = build_maps(CSV)
D = distance_matrix(tuple(idx2name))

train_ds = FathomNetDataset(ROOT/"train_split.csv", IMDIR,
                            name2idx, use_roi=True, split="train")
val_ds   = FathomNetDataset(ROOT/"val_split.csv",   IMDIR,
                            name2idx, use_roi=True, split="val")

train_ds.tfm = build_transforms("train", 384)
val_ds.tfm   = build_transforms("val",   384)
tr_ld = DataLoader(train_ds, BATCH, shuffle=True,
                   num_workers=2, drop_last=True)
va_ld = DataLoader(val_ds, BATCH, shuffle=False, num_workers=2)

mix  = Mixup(mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5,
             switch_prob=0.5, mode="elem", num_classes=79)
focal = FocalLoss(gamma=2.0, alpha=0.5)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type != "cpu"))

# -------------------------------------------------------------------- #
#  TRAIN EACH ARCH                                                     #
# -------------------------------------------------------------------- #
for arch in args.arch:
    print(f"\n================  {arch.upper()}  =================")
    model = ARCH2CLS[arch]().to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = CosineLRScheduler(opt, t_initial=EPOCHS, lr_min=1e-6,
                              warmup_t=2, warmup_lr_init=1e-6)

    best_dist = np.inf; best_path = None

    for epoch in range(1, EPOCHS+1):
        # -------- training ---------
        model.train(); running = 0
        for x,y in tqdm(tr_ld, leave=False, desc=f"{arch}_e{epoch:02d}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            x,oh = mix(x,y)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                log = model(x)["fine"]
                loss = focal(log, oh) + 0.1 * expected_distance(log, y, D)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)

        # -------- validation --------
        model.eval(); hit=tot=dist=0
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type,
                                             dtype=torch.float16):
            for x,y in va_ld:
                x,y = x.to(DEVICE), y.to(DEVICE)
                log = model(x)["fine"]
                pred = log.argmax(1)
                hit  += (pred==y).sum().item()
                tot  += y.size(0)
                dist += expected_distance(log,y,D).item()*x.size(0)
        val_acc, val_dist = hit/tot, dist/tot
        print(f"  Ep{epoch:02d}  Train {running/len(tr_ld.dataset):.3f}  "
              f"ValAcc {val_acc:.3%}  Dist {val_dist:.3f}")

        # keep best checkpoint
        if val_dist < best_dist:
            if best_path: best_path.unlink(missing_ok=True)
            best_dist,val_best = val_dist, val_acc
            best_path = CKDIR/f"{arch}_best.pt"
            torch.save(model.state_dict(), best_path)

        sched.step(epoch)
    print(f"→ {arch}: best dist {best_dist:.3f}  best acc {val_best:.3%}")

print("\n✓ All models trained – best checkpoints in", CKDIR)
