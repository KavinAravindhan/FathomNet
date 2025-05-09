import pandas as pd
import pathlib

test_df = pd.read_csv("fgvc-comp-2025/data/test/annotations.csv", encoding='utf-8-sig')

# If your CSV has a 'path' column instead of 'annotation_id'
test_df["annotation_id"] = test_df["path"].apply(lambda p: int(pathlib.Path(p).stem))

test_ids = set(test_df["annotation_id"].tolist())
print("✅ Ground truth annotation_ids:", len(test_ids))  # Should be 788

sub = pd.read_csv("submission.csv")
submission_ids = set(sub["annotation_id"].tolist())
print("✅ Submission annotation_ids:", len(submission_ids))

missing = test_ids - submission_ids
extra   = submission_ids - test_ids

print("❌ Missing IDs in submission:", missing)
print("❌ Unexpected extra IDs in submission:", extra)
