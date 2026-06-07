"""
Prepare HAM10000 dataset:
  1. Read HAM10000_metadata.csv from HAM10000/
  2. Build full image path index from HAM10000_images_part_1 / part_2
  3. Stratified split into train / val / test (saved as CSVs in data/processed/)

Run once before training:  python prepare_dataset.py
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config, ensure_dirs


LESION_TO_IDX = {name: i for i, name in enumerate(Config.CLASS_NAMES)}


def build_image_index():
    """Map image_id -> absolute path by scanning the two HAM10000 image dirs."""
    index = {}
    for sub in Config.IMAGE_DIRS:
        d = os.path.join(Config.DATA_DIR, sub)
        if not os.path.isdir(d):
            print(f"  [warn] missing image dir: {d}")
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_id = os.path.splitext(fname)[0]
                index[image_id] = os.path.join(d, fname)
    return index


def main():
    ensure_dirs()
    csv_path = os.path.join(Config.DATA_DIR, Config.METADATA_CSV)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] metadata CSV not found at {csv_path}")
        print("        Download HAM10000 from Kaggle / ISIC and place it under HAM10000/")
        sys.exit(1)

    print(f"[1/4] Reading metadata: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      rows: {len(df)}")
    print(f"      columns: {list(df.columns)}")

    print("[2/4] Building image-path index...")
    img_index = build_image_index()
    print(f"      images found on disk: {len(img_index)}")

    df["path"] = df["image_id"].map(img_index)
    missing = df["path"].isna().sum()
    if missing:
        print(f"      [warn] {missing} rows have no matching image — dropping them")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = df["dx"].map(LESION_TO_IDX)
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "dx"].unique()
        print(f"[ERROR] unknown lesion classes in metadata: {bad}")
        sys.exit(1)
    df["label"] = df["label"].astype(int)

    print("[3/4] Class distribution:")
    for cls, idx in LESION_TO_IDX.items():
        n = (df["label"] == idx).sum()
        print(f"      {cls:6s} ({idx}) : {n}")

    print("[4/4] Stratified split...")
    # First carve off test, then val from the remainder
    train_val_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_SPLIT,
        stratify=df["label"],
        random_state=Config.SEED,
    )
    rel_val = Config.VAL_SPLIT / (1.0 - Config.TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=rel_val,
        stratify=train_val_df["label"],
        random_state=Config.SEED,
    )

    out = Config.PROCESSED_DIR
    train_df.to_csv(os.path.join(out, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out, "test.csv"), index=False)

    print(f"      train: {len(train_df)}  |  val: {len(val_df)}  |  test: {len(test_df)}")
    print(f"      written to {out}")
    print("\nDone. You can now run:  python main.py")


if __name__ == "__main__":
    main()
