"""
Usage
-----
python predict.py --ckpts model_checkpoints/ckpt_epoch20.pt \
                  --batch 64 \
                  --out submission.csv
"""
import argparse, pathlib, torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FathomNetDataset, build_transforms
from model   import HierConvNeXt
from taxonomy import build_maps

# --------------------------------------------------------------------------- #
#  Args                                                                       #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--ckpts",  nargs="+", required=True,
                    help="one or more .pt checkpoint paths")
parser.add_argument("--batch",  type=int, default=64)
parser.add_argument("--out",    type=str, default="submission.csv")
parser.add_argument("--size",   type=int, default=384,
                    help="input resolution (use 384 if you fine-tuned)")
args = parser.parse_args()

# --- set device -------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
else:
    DEVICE = torch.device("cpu")

# --------------------------------------------------------------------------- #
#  Load test CSV & label maps                                                 #
# --------------------------------------------------------------------------- #
ROOT = pathlib.Path("fgvc-comp-2025")
TEST_CSV  = ROOT/"data/test/annotations.csv"
TEST_DIR  = ROOT/"data/test"

df_test = pd.read_csv(TEST_CSV)           # has columns: roi_path, annotation_id
train_df = pd.read_csv(ROOT/"data/train/annotations.csv")
name2idx, idx2name = build_maps(train_df)

# --------------------------------------------------------------------------- #
#  Dataset & Loader (no shuffling)                                            #
# --------------------------------------------------------------------------- #
test_ds = FathomNetDataset(TEST_CSV, TEST_DIR, name2idx,
                           use_roi=True, split="val", input_size=args.size)
test_loader = DataLoader(test_ds, args.batch,
                         shuffle=False, num_workers=0, pin_memory=False)

# --------------------------------------------------------------------------- #
#  Build ensemble models                                                      #
# --------------------------------------------------------------------------- #
models = []
for ckpt_path in args.ckpts:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = HierConvNeXt(num_fine=79, num_coarse=len(set(train_df["label"].str.split().str[0])))
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    models.append(model)

# --------------------------------------------------------------------------- #
#  Inference                                                                  #
# --------------------------------------------------------------------------- #
all_preds = []
with torch.no_grad(), torch.autocast(device_type='mps', dtype=torch.float16):
    for imgs, _ in tqdm(test_loader, desc="predict"):
        imgs = imgs.to(DEVICE)
        # ensemble average (logits → probs → mean)
        probs = None
        for m in models:
            logits = m(imgs)["fine"]
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
        probs /= len(models)
        pred_idx = probs.argmax(1).cpu().tolist()
        all_preds.extend(pred_idx)

assert len(all_preds) == len(df_test)

# --------------------------------------------------------------------------- #
#  Build submission DataFrame                                                 #
# --------------------------------------------------------------------------- #
sub = pd.DataFrame({
    "annotation_id": df_test["annotation_id"],
    "concept_name" : [idx2name[i] for i in all_preds]
})

sub.to_csv(args.out, index=False)
print(f"Saved Kaggle submission to {args.out}  ({len(sub)} rows)")

# python predict.py --ckpts model_checkpoints/ckpt_epoch20.pt --out submission.csv

# python predict.py --ckpts ckpt_fold0_ep15.pt ckpt_fold1_ep15.pt \
                #   ckpt_fold2_ep15.pt ckpt_fold3_ep15.pt ckpt_fold4_ep15.pt \
                #   --out submission.csv
