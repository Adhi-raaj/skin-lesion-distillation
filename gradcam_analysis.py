import os
import cv2
import torch
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import Config
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


def preprocess_image(img_path):

    image_bgr = cv2.imread(img_path)

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_rgb = cv2.resize(
        image_rgb,
        (Config.IMG_SIZE, Config.IMG_SIZE)
    )

    image_float = image_rgb.astype(np.float32) / 255.0

    tensor = torch.tensor(image_float).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    return image_float, tensor


def get_target_layer(model, backbone):

    if "efficientnet" in backbone:
        return [model.backbone.conv_head]

    elif "mobilenetv2" in backbone:
        return [model.backbone.conv_head]

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def generate_gradcam(
    model,
    backbone,
    img_path,
    output_path
):

    image_float, tensor = preprocess_image(img_path)

    target_layers = get_target_layer(model, backbone)

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=tensor)[0]

    visualization = show_cam_on_image(
        image_float,
        grayscale_cam,
        use_rgb=True
    )

    Image.fromarray(visualization).save(output_path)

    print(f"Saved -> {output_path}")


def main():

    teacher_path = os.path.join(
        Config.CHECKPOINT_DIR,
        "teacher_baseline_b3.pth"
    )

    student_path = os.path.join(
        Config.DISTILL_DIR,
        "student_mobilenetv2_100_best.pth"
    )

    sample_image = input(
        "Enter full image path: "
    )

    os.makedirs(
        "gradcam_results",
        exist_ok=True
    )

    print("\nLoading teacher...")
    teacher = load_model(
        "efficientnet_b3",
        teacher_path
    )

    print("Loading student...")
    student = load_model(
        "mobilenetv2_100",
        student_path
    )

    print("\nGenerating teacher Grad-CAM...")
    generate_gradcam(
        teacher,
        "efficientnet_b3",
        sample_image,
        "gradcam_results/teacher_gradcam.png"
    )

    print("Generating student Grad-CAM...")
    generate_gradcam(
        student,
        "mobilenetv2_100",
        sample_image,
        "gradcam_results/student_gradcam.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()