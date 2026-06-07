"""
Evaluate and benchmark distilled student models.

Compares:
- Accuracy metrics (accuracy, balanced accuracy, F1)
- Inference speed (ms per image)
- Peak VRAM usage
- Model file size

Saves results to results/distillation_benchmark.json and .csv
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, ensure_dirs
from utils import HAMDataset, get_val_transforms, compute_metrics
from models import build_model


def load_model(checkpoint_path, backbone, config):
    """Load model from checkpoint"""
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(config.DEVICE)


def get_test_loader(config, batch_size=32):
    """Get test dataloader"""
    test_csv = os.path.join(config.PROCESSED_DIR, "test.csv")
    test_ds = HAMDataset(test_csv, transform=get_val_transforms(config.IMG_SIZE))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                     num_workers=4, pin_memory=True)


def benchmark_accuracy(model, test_loader, device, config):
    """Evaluate accuracy metrics"""
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    metrics = compute_metrics(y_true, y_pred, class_names=config.CLASS_NAMES)
    
    return metrics


def benchmark_inference_speed(model, test_loader, device, config, n_runs=100):
    """Measure inference latency"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for imgs, _ in test_loader:
            if len(times) >= n_runs:
                break
            
            imgs = imgs.to(device)
            
            # Warmup
            _ = model(imgs)
            
            # Time
            torch.cuda.synchronize() if "cuda" in str(device) else None
            start = time.time()
            _ = model(imgs)
            torch.cuda.synchronize() if "cuda" in str(device) else None
            end = time.time()
            
            # Per-image latency
            batch_time = (end - start) * 1000  # ms
            per_image_time = batch_time / imgs.shape[0]
            times.extend([per_image_time] * imgs.shape[0])
    
    times = np.array(times[:n_runs])
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }


def benchmark_vram(model, test_loader, device, config):
    """Measure peak VRAM usage"""
    if "cuda" not in str(device):
        return 0.0
    
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            break  # Only need one batch
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    return peak_memory


def get_model_size(checkpoint_path):
    """Get model file size in MB"""
    return os.path.getsize(checkpoint_path) / (1024 ** 2)


def main():
    config = Config
    ensure_dirs()
    
    # Teacher and student checkpoints
    teacher_ckpt = None

    student_ckpts = {
        "mobilenetv2_100": os.path.join(
            config.CHECKPOINT_DIR,
            "distill_T2",
            "student_mobilenetv2_100_best.pth"
        )
    }
    
    # Check which checkpoints exist
    models_to_eval = {}
    
    if teacher_ckpt is not None and os.path.isfile(teacher_ckpt):
        models_to_eval["efficientnet_b3"] = teacher_ckpt
        print(f"✓ Found teacher: {teacher_ckpt}")
    
    for student_name, ckpt_path in student_ckpts.items():
        if os.path.isfile(ckpt_path):
            models_to_eval[student_name] = ckpt_path
            print(f"✓ Found student: {ckpt_path}")
        else:
            print(f"✗ Not found: {ckpt_path}")
    
    if not models_to_eval:
        print("ERROR: No checkpoints found!")
        print(f"Expected teacher at: {teacher_ckpt}")
        print(f"Expected students at: {list(student_ckpts.values())}")
        return
    
    print(f"\nBenchmarking {len(models_to_eval)} models...")
    
    # Load test data
    test_loader = get_test_loader(config, batch_size=32)
    
    results = []
    
    for model_name, ckpt_path in models_to_eval.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        config.BACKBONE = "mobilenetv2_100"
        # Set backbone for loading
        backbone_map = {
            "mobilenetv2_100": "mobilenetv2_100",
        }

        config.BACKBONE = backbone_map[model_name]
        
        # Load model
        print(f"  Loading model...")
        model = load_model(ckpt_path, model_name, config)
        
        # Accuracy
        print(f"  Evaluating accuracy...")
        acc_metrics = benchmark_accuracy(model, test_loader, config.DEVICE, config)
        
        # Speed
        print(f"  Benchmarking inference speed...")
        speed_metrics = benchmark_inference_speed(model, test_loader, config.DEVICE, config, n_runs=100)
        
        # VRAM
        print(f"  Measuring peak VRAM...")
        peak_vram = benchmark_vram(model, test_loader, config.DEVICE, config)
        
        # Model size
        model_size = get_model_size(ckpt_path)
        
        result = {
            "model": model_name,
            "accuracy": acc_metrics["accuracy"],
            "balanced_accuracy": acc_metrics["balanced_accuracy"],
            "f1_macro": acc_metrics["f1_macro"],
            "inference_mean_ms": speed_metrics["mean_ms"],
            "inference_std_ms": speed_metrics["std_ms"],
            "inference_min_ms": speed_metrics["min_ms"],
            "inference_max_ms": speed_metrics["max_ms"],
            "peak_vram_gb": peak_vram,
            "model_size_mb": model_size,
        }
        
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Balanced Acc: {result['balanced_accuracy']:.4f}")
        print(f"    F1 Macro: {result['f1_macro']:.4f}")
        print(f"    Inference: {result['inference_mean_ms']:.2f} ± {result['inference_std_ms']:.2f} ms")
        print(f"    Peak VRAM: {result['peak_vram_gb']:.2f} GB")
        print(f"    Model Size: {result['model_size_mb']:.2f} MB")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    json_path = os.path.join(config.RESULTS_DIR, "distillation_benchmark.json")
    csv_path = os.path.join(config.RESULTS_DIR, "distillation_benchmark.csv")
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    
    # Print summary table
    print(f"\n{results_df.to_string()}")
    
    return results_df


if __name__ == "__main__":
    main()