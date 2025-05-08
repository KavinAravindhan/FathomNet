# fgvc-comp-2025/train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd, pathlib, itertools
from dataset import FathomNetDataset
from taxonomy import build_maps
from model import HierConvNeXt
from timm.data import Mixup
import torch.nn.functional as F
from hier_loss import cached_D, expected_distance

def main():
    ROOT    = pathlib.Path("fgvc-comp-2025")
    CSV     = ROOT/"data/train/annotations.csv"
    IMAGEDIR= ROOT/"data/train"
    MODEL_CHECKPOINT = pathlib.Path("model_checkpoints")
    MODEL_CHECKPOINT.mkdir(exist_ok=True)

    BATCH   = 32
    EPOCHS  = 20
    LR_HEAD = 1e-4
    LR_BODY = 1e-5

    # --- set device -------------------------------------------------------------
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda:1")
    else:
        DEVICE = torch.device("cpu")

    # --- build label maps --------------------------------------------------------
    name2idx, idx2name, coarse_of_idx, coarse_names = build_maps(CSV)
    num_coarse = len(coarse_names)

    # --- split dataframe ---------------------------------------------------------
    df = pd.read_csv(CSV)
    train_idx, val_idx = train_test_split(df.index,
                                        test_size=0.2,
                                        stratify=df["label"],
                                        random_state=42)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val   = df.loc[val_idx].reset_index(drop=True)

    # save splits to avoid leakage
    df_train.to_csv(ROOT/"data/train_split.csv", index=False)
    df_val.to_csv  (ROOT/"data/val_split.csv",   index=False)

    # --- datasets & loaders ------------------------------------------------------
    train_ds = FathomNetDataset(ROOT/"data/train_split.csv", IMAGEDIR,
                                name2idx, use_roi=True, split="train")
    val_ds   = FathomNetDataset(ROOT/"data/val_split.csv",   IMAGEDIR,
                                name2idx, use_roi=True, split="val")

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=2, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=2, pin_memory=False, drop_last=False)
    
    # ------------------------------------------------------------------ #
    #  MixUp / CutMix handler                                            #
    # ------------------------------------------------------------------ #
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.0, num_classes=79)

    # Distance matrix (constant)  ------------------------------------- #
    D = cached_D(tuple(idx2name))

    # --- model -------------------------------------------------------------------
    model = HierConvNeXt(num_fine=79, num_coarse=num_coarse).to(DEVICE)

    # parameter groups: backbone vs heads
    head_params = itertools.chain(model.head_fine.parameters(),
                                model.head_coarse.parameters())
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BODY},
        {"params": head_params,                 "lr": LR_HEAD}
    ])

    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=True)  # works on MPS as of PT2

    # --- training loop -----------------------------------------------------------
    for epoch in range(1, EPOCHS+1):
        model.train(); tot_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"train {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # ---------- MixUp / CutMix ------------------------------------
            imgs, labels_mix = mixup_fn(imgs, labels)
            # labels_mix is float32 one-hot; convert coarse the same way
            coarse_lbl = torch.tensor([coarse_of_idx[l.item()] for l in labels],
                                    device=DEVICE)
            coarse_mix = F.one_hot(coarse_lbl, num_classes=num_coarse).float()
            coarse_mix = coarse_mix * labels_mix.sum(dim=1, keepdim=True)  # preserve λ

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                out = model(imgs)
                loss_fine   = (-labels_mix * F.log_softmax(out["fine"], dim=1)).sum(dim=1).mean()
                loss_coarse = (-coarse_mix * F.log_softmax(out["coarse"], dim=1)).sum(dim=1).mean()
                loss_hier   = expected_distance(out["fine"], labels, D)     # uses original labels
                loss = loss_fine + 0.5*loss_coarse + 0.2*loss_hier
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()*imgs.size(0)

        print(f"Epoch {epoch}  train-loss: {tot_loss/len(train_loader.dataset):.4f}")

        # ---- validation ---------------------------------------------------------
        model.eval(); correct=0; tot=0; dist_sum=0
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)["fine"]
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                tot     += labels.size(0)
                dist_sum += expected_distance(logits, labels, D).item() * imgs.size(0)

        val_acc  = correct/tot
        val_dist = dist_sum/tot
        print(f" val-acc: {val_acc:.3%}   mean-dist: {val_dist:.3f}")


        torch.save(model.state_dict(), MODEL_CHECKPOINT/f"ckpt_epoch{epoch}.pt")

        if epoch == 10:
            train_ds.tfm = build_transforms("train", 384)
            val_ds.tfm   = build_transforms("val",   384)
            train_loader  = DataLoader(train_ds, BATCH//2, shuffle=True,  num_workers=2)
            val_loader    = DataLoader(val_ds,   BATCH//2, shuffle=False, num_workers=2)
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

if __name__ == "__main__":
    main()