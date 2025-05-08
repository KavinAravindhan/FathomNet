# fgvc-comp-2025/taxonomy.py
import pandas as pd, json, pathlib

def build_maps(csv_path: str):
    """
    Returns
    -------
    name_to_idx    : dict  fine-label  ➜ 0‥78
    idx_to_name    : list  index       ➜ fine-label str
    coarse_of_idx  : list  len=79,      value = coarse-id 0‥C-1
    coarse_names   : list  index        ➜ coarse-label str
    """
    df = pd.read_csv(csv_path)
    fine_names = sorted(df["label"].unique().tolist())

    # --- VERY SIMPLE grouping rule -------------------------
    # Use the taxonomic rank *family* (string up to first space)
    # e.g.  "Gadidae Gadus morhua"  -->  "Gadidae"
    #
    # Replace this block later with your own WoRMS lookup table
    def coarse_name(x):
        return x.split()[0]

    coarse_names = sorted({coarse_name(n) for n in fine_names})
    coarse_to_id = {c:i for i,c in enumerate(coarse_names)}

    name_to_idx = {n:i for i,n in enumerate(fine_names)}
    coarse_of_idx = [coarse_to_id[coarse_name(n)] for n in fine_names]

    return name_to_idx, fine_names, coarse_of_idx, coarse_names
