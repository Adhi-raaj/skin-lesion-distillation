"""
Main training entrypoint with support for standard training, distillation, and active learning.

Commands:
  python main.py                           # standard training (auto-resume)
  python main.py --no-resume               # ignore checkpoints, start fresh
  python main.py --mode distillation --student mobilenet_v3_small
  python main.py --mode active_learning --cycles 5
  python main.py --epochs 60 --batch-size 16

Project structure (required):
  training/
    __init__.py (with Trainer, DistillationTrainer, ActiveLearningTrainer)
    trainer.py
    distillation_trainer.py
    active_learning_trainer.py
  
  models/
    __init__.py (with all model utilities)
    auxiliary_modules.py
    ssl_module.py
    gan_module.py
    distillation_module.py
    active_learning.py
  
  utils/
    __init__.py (with dataset and metrics)
    dataset.py
    metrics.py
"""
import argparse
import os
import random
import numpy as np
import torch

from config import Config, ensure_dirs
from training import Trainer, DistillationTrainer, ActiveLearningTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train skin lesion classifier with optional distillation/AL")
    
    # Standard training args
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints, start fresh")
    p.add_argument("--epochs", type=int, default=None, help="override epoch count")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--img-size", type=int, default=None, help="override image size")
    p.add_argument("--backbone", type=str, default=None, help="override backbone model")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    
    # Distillation args
    p.add_argument("--mode", type=str, default="standard", 
                   choices=["standard", "distillation", "active_learning"],
                   help="training mode")
    p.add_argument("--student", type=str, default="mobilenetv3_small_100",
                   choices=["mobilenetv3_small_100","mobilenetv2_100"],
                   help="student backbone for distillation")
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="path to teacher checkpoint for distillation (auto-detected if None)")
    
    # Active Learning args
    p.add_argument("--cycles", type=int, default=5, help="number of AL cycles")
    p.add_argument("--initial-ratio", type=float, default=0.2, help="initial labeled data ratio")
    
    return p.parse_args()


def find_best_checkpoint():
    """Auto-detect teacher checkpoint."""

    best_path = os.path.join(
        Config.CHECKPOINT_DIR,
        "teacher_baseline_b3.pth"
    )

    if os.path.isfile(best_path):
        return best_path

    return None

def main():
    args = parse_args()
    
    # Apply argument overrides
    if args.no_resume:
        Config.RESUME = False
    if args.epochs:
        Config.EPOCHS = args.epochs
        Config.DISTILL_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.img_size:
        Config.IMG_SIZE = args.img_size
    if args.backbone:
        Config.BACKBONE = args.backbone
    if args.lr:
        Config.BASE_LR = args.lr

    ensure_dirs()
    set_seed(Config.SEED)
    
    # Quick GPU sanity check
    if torch.cuda.is_available():
        print(f"[gpu ] {torch.cuda.get_device_name(0)}  "
              f"total VRAM = {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("[gpu ] CUDA not available — training on CPU will be VERY slow.")
    
    # Check required CSVs exist
    need = ["train.csv", "val.csv", "test.csv"]
    if not all(os.path.isfile(os.path.join(Config.PROCESSED_DIR, f)) for f in need):
        print("[error] processed CSVs not found. Run:  python prepare_dataset.py  first.")
        return
    
    print(f"\n[mode] Training mode: {args.mode}")
    Config.show()
    
    # ========== STANDARD TRAINING ==========
    if args.mode == "standard":
        print("\n[train] Standard training on full labeled data...")
        trainer = Trainer(Config)
        trainer.fit()
    
    # ========== DISTILLATION ==========
    elif args.mode == "distillation":
        print(f"\n[distill] Knowledge distillation: EfficientNet-B3 → {args.student}")
        
        # Find teacher checkpoint
        teacher_ckpt = args.teacher_ckpt or find_best_checkpoint()
        if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
            print(f"[error] Teacher checkpoint not found. Please train teacher first with:")
            print(f"        python main.py --epochs 80")
            print(f"        Then run: python main.py --mode distillation --student {args.student}")
            return
        
        print(f"[distill] Teacher checkpoint: {teacher_ckpt}")
        
        Config.STUDENT_BACKBONE = args.student
        Config.DISTILL_BATCH_SIZE = Config.BATCH_SIZE
        
        distill_trainer = DistillationTrainer(Config, teacher_ckpt)
        distill_trainer.fit()
    
    # ========== ACTIVE LEARNING ==========
    elif args.mode == "active_learning":
        print(f"\n[al] Active Learning with {args.cycles} cycles...")
        print(f"[al] Initial labeled ratio: {args.initial_ratio:.1%}")
        
        # Find best model (teacher for uncertainty queries)
        model_ckpt = find_best_checkpoint()
        if not model_ckpt or not os.path.isfile(model_ckpt):
            print(f"[error] Best model checkpoint not found. Please train model first with:")
            print(f"        python main.py --epochs 80")
            return
        
        Config.AL_N_CYCLES = args.cycles
        Config.AL_INITIAL_RATIO = args.initial_ratio
        
        al_trainer = ActiveLearningTrainer(Config, model_ckpt)
        results = al_trainer.run_active_learning_cycles()


if __name__ == "__main__":
    main()
