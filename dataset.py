# fgvc-comp-2025/dataset.py
import pathlib, pandas as pd, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def build_label_maps(df):
    """Return {name:idx}  and  idx→name list for the 79 fine classes."""
    classes = sorted(df["label"].unique().tolist())
    name_to_idx = {c: i for i, c in enumerate(classes)}
    return name_to_idx, classes                 # idx_to_name == classes

def build_transforms(split="train", input_size=224):
    """Basic augmentation for a quick baseline; we’ll extend later."""
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
    else:                                       # val / test
        return T.Compose([
            T.Resize(int(input_size*1.15)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

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
        label = self.name2idx[row["label"]]
        return img, label
