import torch
import onnx
import onnxruntime as ort

from config import Config
from models.auxiliary_modules import RareDiseaseClassifier


MODEL_PATH = "checkpoints/distill_T2/student_mobilenetv2_100_best.pth"
OUTPUT_PATH = "skin_lesion_model.onnx"


def load_student_model():
    model = RareDiseaseClassifier(
        backbone="mobilenetv2_100",
        num_classes=7,
        pretrained=False,
        dropout=0.4,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model


def export_onnx():

    print("Loading model...")

    model = load_student_model()

    dummy_input = torch.randn(
        1,
        3,
        300,
        300
    )

    print("Exporting ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "prediction": {0: "batch_size"}
        }
    )

    print("Validating ONNX...")

    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)

    print("Testing ONNX Runtime...")

    session = ort.InferenceSession(
        OUTPUT_PATH,
        providers=["CPUExecutionProvider"]
    )

    result = session.run(
        None,
        {"image": dummy_input.numpy()}
    )

    print("Output shape:", result[0].shape)

    print("\nSUCCESS")
    print(f"ONNX saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_onnx()