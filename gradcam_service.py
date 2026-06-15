import io
import base64
import numpy as np
import cv2
import torch

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.auxiliary_modules import RareDiseaseClassifier
from config import Config


class GradCAMService:
    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = RareDiseaseClassifier(
            backbone="mobilenetv2_100",
            num_classes=7,
            pretrained=False,
            dropout=0.4
        )

        checkpoint_path = (
            "checkpoints/distill_T2/"
            "student_mobilenetv2_100_best.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )
        self.model.eval()
        self.model.to(self.device)

        self.target_layers = [
            self.model.backbone.conv_head
        ]

    def generate(self, image_bytes):

        pil_img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        original = pil_img.resize((300, 300))
        original_np = np.array(original)

        img = original_np.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        normalized = (img - mean) / std

        tensor = torch.tensor(
            normalized.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

        grayscale_cam = cam(
            input_tensor=tensor
        )[0]

        visualization = show_cam_on_image(
            img,
            grayscale_cam,
            use_rgb=True
        )

        _, buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        encoded = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return encoded