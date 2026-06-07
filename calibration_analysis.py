import os
import numpy as np
import pandas as pd
import torch

from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling

from config import Config
from utils.dataset import HAMDataset, get_val_transforms
from torch.utils.data import DataLoader
from models.auxiliary_modules import RareDiseaseClassifier


DEVICE = torch.device(Config.DEVICE)


def load_model(backbone, ckpt_path):

    model = RareDiseaseClassifier(
        backbone=backbone,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT
    )

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False
    )

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()

    return model


def build_val_loader():

    csv_path = os.path.join(
        Config.PROCESSED_DIR,
        "val.csv"
    )

    ds = HAMDataset(
        csv_path,
        transform=get_val_transforms(Config.IMG_SIZE)
    )

    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    return loader


def collect_probs(model, loader):

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1
            )

            probs_all.append(
                probs.cpu().numpy()
            )

            labels_all.append(
                labels.numpy()
            )

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)

    return probs_all, labels_all


def evaluate(name, probs, labels):

    ece_metric = ECE(15)

    ece = ece_metric.measure(
        probs,
        labels
    )

    preds = np.argmax(
        probs,
        axis=1
    )

    acc = (
        preds == labels
    ).mean()

    print("\n======================")
    print(name)
    print("======================")
    print(f"Accuracy : {acc:.4f}")
    print(f"ECE      : {ece:.4f}")

    return ece


def main():

    teacher_path = os.path.join(
        Config.CHECKPOINT_DIR,
        "teacher_baseline_b3.pth"
    )

    student_path = os.path.join(
        Config.DISTILL_DIR,
        "student_mobilenetv2_100_best.pth"
    )

    loader = build_val_loader()

    print("Loading teacher...")
    teacher = load_model(
        "efficientnet_b3",
        teacher_path
    )

    print("Loading student...")
    student = load_model(
        "mobilenetv2_100",
        student_path
    )

    teacher_probs, labels = collect_probs(
        teacher,
        loader
    )

    student_probs, _ = collect_probs(
        student,
        loader
    )

    teacher_ece = evaluate(
        "Teacher",
        teacher_probs,
        labels
    )

    student_ece = evaluate(
        "Student",
        student_probs,
        labels
    )

    print("\nApplying temperature scaling...")

    scaler = TemperatureScaling()

    scaler.fit(
        student_probs,
        labels
    )

    calibrated_probs = scaler.transform(
        student_probs
    )

    calibrated_ece = evaluate(
        "Student + Temp Scaling",
        calibrated_probs,
        labels
    )

    print("\n======================")
    print("SUMMARY")
    print("======================")
    print(f"Teacher ECE           : {teacher_ece:.4f}")
    print(f"Student ECE           : {student_ece:.4f}")
    print(f"Calibrated Student ECE: {calibrated_ece:.4f}")


if __name__ == "__main__":
    main()