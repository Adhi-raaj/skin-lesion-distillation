"""
Run inference on the test set with optional TTA and write:
  results/test_metrics.json
  results/test_confusion_matrix.png
  results/classification_report.txt

Usage:
  python inference.py                          # uses best_model.pth + TTA
  python inference.py --ckpt checkpoints/epoch_079.pth
  python inference.py --no-tta
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, ensure_dirs
from models import build_model
from utils import HAMDataset, get_val_transforms, get_tta_transforms, compute_metrics


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(Config).to(device)
    # prefer EMA weights if available
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print(f"[load] using EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"[load] using model weights from {ckpt_path}")
    model.eval()
    return model


@torch.no_grad()
def predict_plain(model, loader, device):
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_p.append(probs)
        all_y.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


@torch.no_grad()
def predict_tta(model, csv_path, img_size, batch_size, device, num_workers):
    """Run inference once per TTA transform on the SAME test set, average softmax."""
    transforms_list = get_tta_transforms(img_size)
    sum_probs = None
    y_true = None
    for i, tfm in enumerate(transforms_list):
        ds = HAMDataset(csv_path, transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        probs, labels = predict_plain(model, loader, device)
        sum_probs = probs if sum_probs is None else sum_probs + probs
        y_true = labels
        print(f"  [tta] pass {i+1}/{len(transforms_list)}  acc so far = "
              f"{(sum_probs.argmax(axis=1) == y_true).mean():.4f}")
    return sum_probs / len(transforms_list), y_true


def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="path to checkpoint; defaults to checkpoints/best_model.pth")
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    device = torch.device(Config.DEVICE)

    ckpt = args.ckpt or os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = load_model(ckpt, device)

    test_csv = os.path.join(Config.PROCESSED_DIR, "test.csv")
    use_tta = Config.USE_TTA and not args.no_tta

    if use_tta:
        print("[infer] running with TTA...")
        probs, y_true = predict_tta(model, test_csv, Config.IMG_SIZE,
                                    Config.BATCH_SIZE, device, Config.NUM_WORKERS)
    else:
        print("[infer] running plain inference...")
        ds = HAMDataset(test_csv, transform=get_val_transforms(Config.IMG_SIZE))
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
        probs, y_true = predict_plain(model, loader, device)

    y_pred = probs.argmax(axis=1)
    m = compute_metrics(y_true, y_pred, class_names=Config.CLASS_NAMES)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"accuracy           : {m['accuracy']:.4f}")
    print(f"balanced accuracy  : {m['balanced_accuracy']:.4f}")
    print(f"f1 (macro)         : {m['f1_macro']:.4f}")
    print(f"f1 (weighted)      : {m['f1_weighted']:.4f}")
    print()
    print(m["classification_report"])

    # save
    out = {
        "accuracy": float(m["accuracy"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "f1_macro": float(m["f1_macro"]),
        "f1_weighted": float(m["f1_weighted"]),
        "per_class_precision": m["per_class_precision"].tolist(),
        "per_class_recall": m["per_class_recall"].tolist(),
        "per_class_f1": m["per_class_f1"].tolist(),
        "tta": use_tta,
        "ckpt": ckpt,
    }
    with open(os.path.join(Config.RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(Config.RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(m["classification_report"])
    plot_confusion(m["confusion_matrix"], Config.CLASS_NAMES,
                   os.path.join(Config.RESULTS_DIR, "test_confusion_matrix.png"))
    print(f"\n[saved] results -> {Config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
