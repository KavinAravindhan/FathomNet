import argparse, pathlib, torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FathomNetDataset, build_transforms
from model   import HierConvNeXt
from taxonomy import build_maps

parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--out", type=str, default="submission.csv")
parser.add_argument("--size", type=int, default=384)
args = parser.parse_args()

# Set device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# Load label mappings
ROOT = pathlib.Path("fgvc-comp-2025")
TRAIN_CSV = ROOT / "data/train/annotations.csv"
TEST_CSV = ROOT / "data/test/annotations.csv"
TEST_DIR = ROOT / "data/test"

name2idx, idx2name, _, _ = build_maps(TRAIN_CSV)
train_df = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Extract annotation_id from path
df_test["annotation_id"] = df_test["path"].apply(
    lambda p: int(pathlib.Path(p).stem.split("_")[-1])
)

# Dataset and loader
test_ds = FathomNetDataset(TEST_CSV, TEST_DIR, name2idx, use_roi=True,
                           split="val", input_size=args.size)
test_loader = DataLoader(test_ds, args.batch, shuffle=False, num_workers=0, pin_memory=False)

# Load ensemble models
models = []
for ckpt_path in args.ckpts:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = HierConvNeXt(num_fine=79, num_coarse=len(set(train_df["label"].str.split().str[0])))
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    models.append(model)

# Inference
all_preds = []
with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
    # for imgs, _ in tqdm(test_loader, desc="predict"):
    for imgs in tqdm(test_loader, desc="predict"):
        imgs = imgs.to(DEVICE)
        probs = None
        for model in models:
            logits = model(imgs)["fine"]
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
        probs /= len(models)
        pred_idx = probs.argmax(1).cpu().tolist()
        all_preds.extend(pred_idx)

# Build submission
submission = pd.DataFrame({
    "annotation_id": df_test["annotation_id"],
    "concept_name": [idx2name[i] for i in all_preds]
})
submission = submission.sort_values("annotation_id")

# Validate concept_names
valid_names = set(train_df["label"].unique())
invalid = set(submission["concept_name"]) - valid_names
if invalid:
    print("❌ Invalid concept_name(s) detected in submission:")
    for i in invalid:
        print(" -", i)
    raise ValueError("Submission contains invalid concept_name(s). Please fix the predictions.")
else:
    print("✅ All concept_name values are valid.")

submission.to_csv(args.out, index=False)
print(f"✅ Submission saved to {args.out} with {len(submission)} entries.")

# python predict.py --ckpts model_checkpoints/hier_loss/ckpt_epoch20.pt --out submission.csv

# python predict.py \
#   --ckpts model_checkpoints/ckpt_epoch18.pt model_checkpoints/ckpt_epoch19.pt model_checkpoints/ckpt_epoch20.pt \
#   --out submission.csv