import pandas as pd, pathlib

csv_path = "fgvc-comp-2025/data/train/annotations.csv"
root     = pathlib.Path(csv_path).parent

df = pd.read_csv(csv_path)

print("train rows :", len(df))                       # 23 699
print("unique classes :", df['label'].nunique())     # 79
print("files in images :", len(list((root/'images').glob('*'))))
print("files in rois   :", len(list((root/'rois').glob('*'))))
print(df.groupby('label').size().describe())
