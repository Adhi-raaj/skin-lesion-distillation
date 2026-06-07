"""
HAM10000 PyTorch Dataset with strong Albumentations augmentation.
"""
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet stats (we use pretrained backbones)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        A.Rotate(
            limit=25,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),

        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int):
    """Light, deterministic augmentations for test-time averaging."""
    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(90, 90), p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(180, 180), p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(270, 270), p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
                   A.CenterCrop(img_size, img_size), A.Normalize(MEAN, STD), ToTensorV2()]),
        A.Compose([A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
                   A.CenterCrop(img_size, img_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(MEAN, STD), ToTensorV2()]),
    ]


class HAMDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        # cache as numpy for slight speed boost
        self.paths = self.df["path"].values
        self.labels = self.df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return self.labels
