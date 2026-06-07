"""
Knowledge Distillation Trainer with full checkpoint support.

Trains a lightweight student model using guidance from a pre-trained teacher model.
The student learns both from hard labels (ground truth) and soft targets (teacher output).

Key features:
- Saves checkpoint EVERY epoch → can crash & resume any time
- Auto-resumes from latest checkpoint if cfg.RESUME is True
- Keeps only the last N epoch checkpoints + best model
- Mixed precision (AMP)
- Cosine LR schedule with linear warmup
- EMA model maintained alongside the live model
"""
import os
import json
import math
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from utils.dataset import HAMDataset, get_train_transforms, get_val_transforms
from utils.metrics import compute_metrics, AverageMeter
from models.auxiliary_modules import build_model, RareDiseaseClassifier
from models.ssl_module import build_criterion, mixup_data, cutmix_data, mixed_criterion
from models.gan_module import ModelEMA
from models.distillation_module import DistillationLoss, DistillationMetrics


def _class_weights_from_labels(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _make_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def _cosine_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


class DistillationTrainer:
    """
    Trainer for knowledge distillation with checkpoint support.
    
    Loads a pre-trained teacher model and trains a lightweight student
    using the teacher's soft targets as guidance.
    
    Features:
    - Per-epoch checkpointing (resume on crash)
    - Best model tracking
    - Distillation metrics (similarity, agreement)
    """
    
    def __init__(self, cfg, teacher_checkpoint_path):
        """
        Args:
            cfg: Config object
            teacher_checkpoint_path: path to pre-trained teacher checkpoint
        """
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        if cfg.BENCHMARK and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        print(f"[distill] Loading teacher from: {teacher_checkpoint_path}")
        
        # ---- Load TEACHER (frozen) ----
        self.teacher = RareDiseaseClassifier(
            backbone=cfg.TEACHER_BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            pretrained=False,
            dropout=cfg.DROPOUT,
        ).to(self.device)
        teacher_ckpt = torch.load(teacher_checkpoint_path, map_location=self.device, weights_only=False)
        self.teacher.load_state_dict(teacher_ckpt["model"])
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print(f"[distill] Teacher loaded (frozen). VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # ---- Load DATA ----
        train_csv = os.path.join(cfg.PROCESSED_DIR, "train.csv")
        val_csv = os.path.join(cfg.PROCESSED_DIR, "val.csv")
        assert os.path.isfile(train_csv), f"missing {train_csv}. Run prepare_dataset.py first."
        assert os.path.isfile(val_csv), f"missing {val_csv}."
        
        self.train_ds = HAMDataset(train_csv, transform=get_train_transforms(cfg.IMG_SIZE))
        self.val_ds = HAMDataset(val_csv, transform=get_val_transforms(cfg.IMG_SIZE))
        
        labels = self.train_ds.get_labels()
        sampler = _make_weighted_sampler(labels, cfg.NUM_CLASSES) if cfg.USE_WEIGHTED_SAMPLER else None
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.DISTILL_BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.DISTILL_BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )
        
        # ---- Build STUDENT (trainable) ----
        print(f"[distill] Building student: {cfg.STUDENT_BACKBONE}")
        # Create student with correct backbone
        self.model = RareDiseaseClassifier(
            backbone=cfg.STUDENT_BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            pretrained=True,
            dropout=cfg.DROPOUT,
        ).to(self.device)
        #elf.ema = ModelEMA(self.model, decay=0.9995)
        print(f"[distill] Student built. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # ---- Loss / Optimizer / Scheduler ----
        class_w = _class_weights_from_labels(labels, cfg.NUM_CLASSES).to(self.device)
        self.criterion = DistillationLoss(
            temperature=cfg.DISTILL_TEMPERATURE,
            alpha=cfg.DISTILL_ALPHA
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.DISTILL_BASE_LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scaler = GradScaler(enabled=cfg.USE_AMP and self.device.type == "cuda")
        
        # ---- State ----
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.history = []
        
        # Create distill checkpoint directory
        os.makedirs(cfg.DISTILL_DIR, exist_ok=True)
        
        # Auto-resume if enabled (only if checkpoint is compatible)
        if cfg.RESUME:
            self._try_resume()
    
    # --------------------------------------------------------------------- #
    #  checkpoint I/O                                                       #
    # --------------------------------------------------------------------- #
    
    def _ckpt_path(self, epoch):
        return os.path.join(self.cfg.DISTILL_DIR, 
                           f"student_{self.cfg.STUDENT_BACKBONE}_epoch_{epoch:03d}.pth")
    
    def _best_path(self):
        return os.path.join(self.cfg.DISTILL_DIR, 
                           f"student_{self.cfg.STUDENT_BACKBONE}_best.pth")
    
    def _latest_path(self):
        files = sorted(glob.glob(os.path.join(
            self.cfg.DISTILL_DIR,
            f"student_{self.cfg.STUDENT_BACKBONE}_epoch_*.pth"
        )))
        return files[-1] if files else None
    
    def _save(self, epoch, val_metrics):
        """Save per-epoch checkpoint and track best model"""
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            #ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "student_backbone": self.cfg.STUDENT_BACKBONE,
            "config": {k: v for k, v in vars(self.cfg).items()
                      if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))},
        }
        
        # Per-epoch checkpoint (always save)
        if self.cfg.SAVE_EVERY_EPOCH:
            torch.save(state, self._ckpt_path(epoch))
            
            # Rotate: keep only last N
            files = sorted(glob.glob(os.path.join(
                self.cfg.DISTILL_DIR,
                f"student_{self.cfg.STUDENT_BACKBONE}_epoch_*.pth"
            )))
            for f in files[:-self.cfg.KEEP_LAST_N]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        
        # Best checkpoint
        if self.cfg.SAVE_BEST and val_metrics["accuracy"] > self.best_val_acc:
            self.best_val_acc = val_metrics["accuracy"]
            state["best_val_acc"] = self.best_val_acc
            torch.save(state, self._best_path())
            print(f"      [save] NEW BEST STUDENT -> acc={self.best_val_acc:.4f}")
    
    def _try_resume(self):
        """Auto-resume from latest checkpoint if it exists and is compatible"""
        path = self._latest_path()
        if path is None:
            print("[resume] no checkpoint found, starting fresh")
            return
        
        print(f"[resume] found checkpoint: {os.path.basename(path)}")
        
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            
            # Check if checkpoint is for the same student backbone
            ckpt_backbone = ckpt.get("student_backbone", "unknown")
            if ckpt_backbone != self.cfg.STUDENT_BACKBONE:
                print(f"[resume] checkpoint is for {ckpt_backbone}, but current is {self.cfg.STUDENT_BACKBONE}")
                print(f"[resume] skipping incompatible checkpoint, starting fresh")
                return
            
            print(f"[resume] loading checkpoint (backbone: {ckpt_backbone})")
            self.model.load_state_dict(ckpt["model"])
            #elf.ema.load_state_dict(ckpt["ema"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
            
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_acc = ckpt.get("best_val_acc", 0.0)
            self.history = ckpt.get("history", [])
            
            print(f"[resume] resuming at epoch {self.start_epoch}, best_val_acc={self.best_val_acc:.4f}")
            
        except Exception as e:
            print(f"[resume] error loading checkpoint: {e}")
            print(f"[resume] starting fresh instead")
    
    # --------------------------------------------------------------------- #
    #  train / eval loops                                                   #
    # --------------------------------------------------------------------- #
    
    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_m = AverageMeter()
        acc_m = AverageMeter()
        similarity_m = AverageMeter()
        agreement_m = AverageMeter()
        t0 = time.time()
        
        lr = _cosine_lr(epoch, self.cfg.DISTILL_EPOCHS, self.cfg.WARMUP_EPOCHS,
                        self.cfg.DISTILL_BASE_LR, self.cfg.MIN_LR)
        _set_lr(self.optimizer, lr)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for step, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Get teacher logits (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(imgs)
            
            # Get student logits and compute loss
            with autocast(device_type=self.device.type, enabled=self.cfg.USE_AMP):
                student_logits = self.model(imgs)
                loss = self.criterion(student_logits, teacher_logits, labels)
                loss = loss / self.cfg.GRAD_ACCUM_STEPS
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.cfg.GRAD_ACCUM_STEPS == 0:
                if self.cfg.GRAD_CLIP and self.cfg.GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                #elf.ema.update(self.model)
            
            with torch.no_grad():
                preds = student_logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                
                # Distillation-specific metrics
                similarity = DistillationMetrics.compute_logit_similarity(student_logits, teacher_logits)
                agreement = DistillationMetrics.compute_agreement(student_logits, teacher_logits)
            
            loss_m.update(loss.item() * self.cfg.GRAD_ACCUM_STEPS, imgs.size(0))
            acc_m.update(acc, imgs.size(0))
            similarity_m.update(similarity, imgs.size(0))
            agreement_m.update(agreement, imgs.size(0))
            
            if step % 50 == 0:
                print(f"   ep{epoch:03d}  step {step:4d}/{len(self.train_loader)}  "
                      f"lr={lr:.2e}  loss={loss_m.avg:.4f}  acc~{acc_m.avg:.4f}  "
                      f"sim={similarity_m.avg:.4f}  agree={agreement_m.avg:.4f}")
        
        dt = time.time() - t0
        print(f"   [train] epoch {epoch} done in {dt/60:.1f} min  loss={loss_m.avg:.4f}  "
              f"acc~{acc_m.avg:.4f}  similarity={similarity_m.avg:.4f}  agreement={agreement_m.avg:.4f}")
        
        return {
            "train_loss": loss_m.avg,
            "train_acc": acc_m.avg,
            "train_similarity": similarity_m.avg,
            "train_agreement": agreement_m.avg,
            "lr": lr
        }
    
    @torch.no_grad()
    def _evaluate(self, loader, model=None):
        model = model or self.model
        model.eval()
        all_preds, all_labels = [], []
        
        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.cfg.USE_AMP):
                logits = model(imgs)
            
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        m = compute_metrics(y_true, y_pred, class_names=self.cfg.CLASS_NAMES)
        
        return m
    
    # --------------------------------------------------------------------- #
    #  main entry                                                           #
    # --------------------------------------------------------------------- #
    
    def fit(self):
        cfg = self.cfg
        
        for epoch in range(self.start_epoch, cfg.DISTILL_EPOCHS):
            print(f"\n==== Distillation Epoch {epoch+1}/{cfg.DISTILL_EPOCHS} ====")
            print(f"Student: {cfg.STUDENT_BACKBONE}")
            
            tr = self._train_one_epoch(epoch)
            
            print(f"   [val ] evaluating...")
            val_metrics = self._evaluate(self.val_loader, self.model)
            
            print(f"   [val ] acc={val_metrics['accuracy']:.4f}  bal_acc={val_metrics['balanced_accuracy']:.4f}")
            
            row = {
                "epoch": epoch,
                **tr,
                "val_acc": val_metrics["accuracy"],
                "val_bal_acc": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1_macro"],
            }
            self.history.append(row)
            
            self._save(epoch, val_metrics)
        
        print(f"\n[done] Distillation training complete!")
        print(f"[done] Best student accuracy: {self.best_val_acc:.4f}")
        print(f"[done] Best checkpoint: {self._best_path()}")
        
        return self.history
