import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve

from config import Config
from utils import HAMDataset, get_val_transforms
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

    model.load_state_dict(ckpt["model"])

    model.to(DEVICE)
    model.eval()

    return model


def build_loader():

    csv_path = os.path.join(
        Config.PROCESSED_DIR,
        "test.csv"
    )

    ds = HAMDataset(
        csv_path,
        transform=get_val_transforms(Config.IMG_SIZE)
    )

    return DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )


def collect_confidences(model, loader):

    confs = []
    correct = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(DEVICE)

            logits = model(imgs)

            probs = torch.softmax(
                logits,
                dim=1
            )

            confidence, preds = probs.max(dim=1)

            confs.extend(
                confidence.cpu().numpy()
            )

            correct.extend(
                (preds.cpu() == labels).numpy()
            )

    return np.array(confs), np.array(correct)


def main():
    
    teacher_ckpt = os.path.join(
    Config.CHECKPOINT_DIR,
    "teacher_baseline_b3.pth"
)

    student_ckpt = os.path.join(
        Config.CHECKPOINT_DIR,
        "distill_T2",
        "student_mobilenetv2_100_best.pth"
    )

    loader = build_loader()

    print("Loading teacher...")
    teacher = load_model(
        "efficientnet_b3",
        teacher_ckpt
    )

    print("Loading student...")
    student = load_model(
        "mobilenetv2_100",
        student_ckpt
    )

    teacher_conf, teacher_corr = collect_confidences(
        teacher,
        loader
    )

    student_conf, student_corr = collect_confidences(
        student,
        loader
    )

    teacher_true, teacher_pred = calibration_curve(
        teacher_corr,
        teacher_conf,
        n_bins=10
    )

    student_true, student_pred = calibration_curve(
        student_corr,
        student_conf,
        n_bins=10
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect Calibration"
    )

    plt.plot(
        teacher_pred,
        teacher_true,
        marker="o",
        label="Teacher"
    )

    plt.plot(
        student_pred,
        student_true,
        marker="s",
        label="Student T=2"
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    plt.legend()
    plt.grid(True)

    os.makedirs(
        "results",
        exist_ok=True
    )

    save_path = os.path.join(
        "results",
        "reliability_diagram.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()