"""
Test-Time Augmentation (TTA).
Filename kept as few_shot_module.py for compatibility with original layout.

TTA: run the model on several augmented versions of the same image and
average the softmax outputs. Usually adds 0.5-1.5% accuracy "for free".
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def tta_predict(model, image_tensor_list, device):
    """
    image_tensor_list: list of (B, C, H, W) tensors (same B), each from
                       a different TTA transform of the SAME images.
    Returns averaged softmax probabilities (B, num_classes).
    """
    model.eval()
    probs_sum = None
    for x in image_tensor_list:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else (probs_sum + probs)
    return probs_sum / len(image_tensor_list)
