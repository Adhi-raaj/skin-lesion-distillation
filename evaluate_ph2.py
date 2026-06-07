import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import Config
from models.auxiliary_modules import RareDiseaseClassifier


DEVICE = torch.device(Config.DEVICE)


class PH2Dataset(Dataset):

    def __init__(self, excel_path, image_root):

        self.samples = []

        df = pd.read_excel(
            excel_path,
            header=12
        )
        print(df.columns.tolist())

        for _, row in df.iterrows():

            image_id = str(row["Image Name"]).strip()

            if pd.notna(row["Melanoma"]):
                label = 1

            elif (
                pd.notna(row["Common Nevus"])
                or pd.notna(row["Atypical Nevus"])
            ):
                label = 0

            else:
                continue

            img_path = os.path.join(
                image_root,
                image_id,
                f"{image_id}_Dermoscopic_Image",
                f"{image_id}.bmp"
            )

            if os.path.isfile(img_path):
                self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label


def load_student():

    model = RareDiseaseClassifier(
        backbone="mobilenetv2_100",
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT
    )

    ckpt = torch.load(
        os.path.join(
            Config.CHECKPOINT_DIR,
            "distill_T2",
            "student_mobilenetv2_100_best.pth"
        ),
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(ckpt["model"])

    model.to(DEVICE)
    model.eval()

    return model


def main():

    excel_path = r"PH2Dataset\PH2_dataset.xlsx"

    image_root = r"PH2Dataset\PH2 Dataset images"

    dataset = PH2Dataset(
        excel_path,
        image_root
    )

    print(f"Loaded {len(dataset)} PH2 images")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    model = load_student()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(DEVICE)

            logits = model(imgs)

            preds_7class = logits.argmax(dim=1)

            preds_binary = []

            for p in preds_7class.cpu().numpy():

                if p == 4:
                    preds_binary.append(1)

                elif p == 5:
                    preds_binary.append(0)

                else:
                    preds_binary.append(0)

            y_pred.extend(preds_binary)
            y_true.extend(labels.numpy())

    print("\n===== PH2 EXTERNAL VALIDATION =====\n")

    print(
        "Accuracy:",
        round(accuracy_score(y_true, y_pred), 4)
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(y_true, y_pred),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(y_true, y_pred),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(y_true, y_pred),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(y_true, y_pred),
            4
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(y_true, y_pred)
    )


if __name__ == "__main__":
    main()