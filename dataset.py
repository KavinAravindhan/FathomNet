# fgvc-comp-2025/dataset.py
import pathlib, pandas as pd, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from timm.data import create_transform

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def build_label_maps(df):
    """Return {name:idx}  and  idx→name list for the 79 fine classes."""
    classes = sorted(df["label"].unique().tolist())
    name_to_idx = {c: i for i, c in enumerate(classes)}
    return name_to_idx, classes                 # idx_to_name == classes

# --------------------------------------------------------------------------- #
#  Transforms                                                                 #
# --------------------------------------------------------------------------- #
def build_transforms(split="train", input_size=224):
    """
    Stronger Augment:
      • RandAugment (N=2, M=9)           – timm default
      • Random Erasing (p=0.25)
    Validation/test: just Resize + CenterCrop.
    """
    if split == "train":
        return create_transform(
            input_size=input_size,
            is_training=True,
            no_aug=False,          # enable RandAugment
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            hflip=0.5,
            vflip=0.0,
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            re_prob=0.25,          # Random Erase prob
            re_mode='pixel',
            re_count=1,
        )
    else:  # val / test
        return create_transform(
            input_size=input_size,
            is_training=False,
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )


# --------------------------------------------------------------------------- #
#  Main Dataset                                                               #
# --------------------------------------------------------------------------- #

class FathomNetDataset(Dataset):
    """
    Parameters
    ----------
    csv_file : path to annotations.csv  (must have columns [`file_name`,
               `annotation_id`, `label`])
    root_dir : folder that contains `images/` and `rois/`
    name_to_idx : dict from label string → integer id
    use_roi : True = load cropped organism chip; False = load full frame
    split : 'train' or 'val'  (decides augmentation)
    """
    def __init__(self, csv_file, root_dir, name_to_idx,
                 use_roi=True, split="train", input_size=224):
        self.df = pd.read_csv(csv_file)
        self.root_dir  = pathlib.Path(root_dir)
        self.use_roi   = use_roi
        self.name2idx  = name_to_idx
        self.tfm       = build_transforms(split, input_size)

        
        # ------------------------------------------------------------------ #
        # Decide which column already stores the image path.                 #
        # ------------------------------------------------------------------ #
        if use_roi:
            if "roi_path" in self.df.columns:
                self.path_col = "roi_path"
            elif "path" in self.df.columns:            # fallback (older script)
                self.path_col = "path"
            else:
                # build it from annotation_id
                assert "annotation_id" in self.df.columns,\
                       "Need roi_path, path, or annotation_id in CSV"
                self.df["roi_path"] = (
                    self.root_dir / "rois" /
                    (self.df["annotation_id"].astype(str) + ".png")
                )
                self.path_col = "roi_path"
        else:   # full frame
            if "image_path" in self.df.columns:
                self.path_col = "image_path"
            elif "file_name" in self.df.columns:
                self.df["image_path"] = (
                    self.root_dir / "images" / self.df["file_name"]
                )
                self.path_col = "image_path"
            else:
                raise ValueError("CSV must contain image_path or file_name")


    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.path_col]).convert("RGB")
        img = self.tfm(img)
        if "label" in row and pd.notna(row["label"]):
            label = self.name2idx[row["label"]]
            return img, label
        else:
            return img
