# fgvc-comp-2025/train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd, pathlib, itertools
from dataset import FathomNetDataset
from taxonomy import build_maps
from model import HierConvNeXt

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
    DEVICE  = torch.device("mps")   # Apple GPU

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
                          num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=2, pin_memory=False)


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
            coarse_lbl = torch.tensor([coarse_of_idx[l] for l in labels],
                                    device=DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type='mps', dtype=torch.float16):
                out = model(imgs)
                loss_fine   = criterion(out["fine"],   labels)
                loss_coarse = criterion(out["coarse"], coarse_lbl)
                loss = loss_fine + 0.5*loss_coarse
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()*imgs.size(0)

        print(f"Epoch {epoch}  train-loss: {tot_loss/len(train_loader.dataset):.4f}")

        # ---- validation ---------------------------------------------------------
        model.eval(); correct=0; tot=0
        with torch.no_grad(), torch.autocast(device_type='mps', dtype=torch.float16):
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)["fine"]
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                tot     += labels.size(0)
        print(f" val-acc: {correct/tot:.3%}")

        torch.save(model.state_dict(), MODEL_CHECKPOINT/f"ckpt_epoch{epoch}.pt")

if __name__ == "__main__":
    main()