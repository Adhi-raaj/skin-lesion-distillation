"""
Trainer.

Key properties:
  * Saves a checkpoint EVERY epoch -> can crash & resume any time
  * Auto-resumes from latest checkpoint if cfg.RESUME is True
  * Keeps only the last N epoch checkpoints + best model
  * Mixed precision (AMP) — needed to fit batch=32 at 300px on 12GB
  * Cosine LR schedule with linear warmup
  * MixUp / CutMix applied stochastically
  * EMA model maintained alongside the live model
  * Gradient clipping, gradient accumulation
"""
import os
import json
import math
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

# Direct imports from submodules
from utils.dataset import HAMDataset, get_train_transforms, get_val_transforms
from utils.metrics import compute_metrics, AverageMeter
from models.auxiliary_modules import build_model
from models.ssl_module import build_criterion, mixup_data, cutmix_data, mixed_criterion
from models.gan_module import ModelEMA


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  trainer                                                                    #
# --------------------------------------------------------------------------- #

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        if cfg.BENCHMARK and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # ---- data ----
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
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )

        # ---- model ----
        self.model = build_model(cfg).to(self.device)
        self.ema = ModelEMA(self.model, decay=0.9995)

        # ---- loss / opt / sched ----
        class_w = _class_weights_from_labels(labels, cfg.NUM_CLASSES).to(self.device)
        self.criterion = build_criterion(cfg, class_weights=class_w)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.BASE_LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scaler = GradScaler(enabled=cfg.USE_AMP and self.device.type == "cuda")

        # ---- state ----
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.history = []

        # auto-resume
        if cfg.RESUME:
            self._try_resume()

    # --------------------------------------------------------------------- #
    #  checkpoint I/O                                                       #
    # --------------------------------------------------------------------- #
    def _ckpt_path(self, epoch):
        return os.path.join(self.cfg.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")

    def _best_path(self):
        return os.path.join(self.cfg.CHECKPOINT_DIR, "best_model.pth")

    def _latest_path(self):
        files = sorted(glob.glob(os.path.join(self.cfg.CHECKPOINT_DIR, "epoch_*.pth")))
        return files[-1] if files else None

    def _save(self, epoch, val_metrics):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": {k: v for k, v in vars(self.cfg).items()
                       if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))},
        }
        # per-epoch
        if self.cfg.SAVE_EVERY_EPOCH:
            torch.save(state, self._ckpt_path(epoch))
            # rotate old
            files = sorted(glob.glob(os.path.join(self.cfg.CHECKPOINT_DIR, "epoch_*.pth")))
            for f in files[:-self.cfg.KEEP_LAST_N]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        # best
        if self.cfg.SAVE_BEST and val_metrics["accuracy"] > self.best_val_acc:
            self.best_val_acc = val_metrics["accuracy"]
            state["best_val_acc"] = self.best_val_acc
            torch.save(state, self._best_path())
            print(f"      [save] new best -> {self._best_path()}  acc={self.best_val_acc:.4f}")

    def _try_resume(self):

    # Skip resume during Active Learning
        if getattr(self.cfg, "USE_ACTIVE_LEARNING", False):
            print("[resume] skipped during active learning")
            return

        path = self._latest_path()
        if path is None:
            print("[resume] no checkpoint found, starting fresh")
            return
        print(f"[resume] loading {path}")
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        try:
            self.scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        self.history = ckpt.get("history", [])
        print(
            f"[resume] resuming at epoch {self.start_epoch}, "
            f"best_val_acc={self.best_val_acc:.4f}"
        )

    # --------------------------------------------------------------------- #
    #  train / eval loops                                                   #
    # --------------------------------------------------------------------- #
    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_m = AverageMeter()
        acc_m = AverageMeter()
        t0 = time.time()

        lr = _cosine_lr(epoch, self.cfg.EPOCHS, self.cfg.WARMUP_EPOCHS,
                        self.cfg.BASE_LR, self.cfg.MIN_LR)
        _set_lr(self.optimizer, lr)

        self.optimizer.zero_grad(set_to_none=True)
        for step, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # mixup / cutmix
            use_mix = (np.random.rand() < self.cfg.MIXUP_PROB) and (
                self.cfg.USE_MIXUP or self.cfg.USE_CUTMIX
            )
            if use_mix:
                if self.cfg.USE_MIXUP and self.cfg.USE_CUTMIX:
                    choice = "mixup" if np.random.rand() < 0.5 else "cutmix"
                elif self.cfg.USE_MIXUP:
                    choice = "mixup"
                else:
                    choice = "cutmix"
                if choice == "mixup":
                    imgs, y_a, y_b, lam = mixup_data(imgs, labels, self.cfg.MIXUP_ALPHA)
                else:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, labels, self.cfg.CUTMIX_ALPHA)
            else:
                y_a, y_b, lam = labels, labels, 1.0

            with autocast(device_type=self.device.type, enabled=self.cfg.USE_AMP):
                logits = self.model(imgs)
                loss = mixed_criterion(self.criterion, logits, y_a, y_b, lam)
                loss = loss / self.cfg.GRAD_ACCUM_STEPS

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.GRAD_ACCUM_STEPS == 0:
                if self.cfg.GRAD_CLIP and self.cfg.GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                # accuracy here is for the "primary" labels (y_a) — approx only when mixed
                acc = (preds == y_a).float().mean().item()

            loss_m.update(loss.item() * self.cfg.GRAD_ACCUM_STEPS, imgs.size(0))
            acc_m.update(acc, imgs.size(0))

            if step % 50 == 0:
                print(f"   ep{epoch:03d}  step {step:4d}/{len(self.train_loader)}  "
                      f"lr={lr:.2e}  loss={loss_m.avg:.4f}  acc~{acc_m.avg:.4f}")

        dt = time.time() - t0
        print(f"   [train] epoch {epoch} done in {dt/60:.1f} min  loss={loss_m.avg:.4f}  acc~{acc_m.avg:.4f}")
        return {"train_loss": loss_m.avg, "train_acc": acc_m.avg, "lr": lr}

    @torch.no_grad()
    def _evaluate(self, loader, model=None):
        model = model or self.model
        model.eval()
        all_preds, all_labels = [], []
        loss_m = AverageMeter()
        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type, enabled=self.cfg.USE_AMP):
                logits = model(imgs)
                loss = self.criterion(logits, labels)
            loss_m.update(loss.item(), imgs.size(0))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        m = compute_metrics(y_true, y_pred, class_names=self.cfg.CLASS_NAMES)
        m["loss"] = loss_m.avg
        return m

    # --------------------------------------------------------------------- #
    #  main entry                                                           #
    # --------------------------------------------------------------------- #
    def fit(self):
        cfg = self.cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

        for epoch in range(self.start_epoch, cfg.EPOCHS):
            print(f"\n==== Epoch {epoch+1}/{cfg.EPOCHS} ====")
            tr = self._train_one_epoch(epoch)
            print(f"   [val ] evaluating live model...")
            val_live = self._evaluate(self.val_loader, self.model)
            print(f"   [val ] evaluating EMA model...")
            val_ema = self._evaluate(self.val_loader, self.ema.module)
            # pick the better of live vs EMA for "best" tracking
            val_metrics = val_ema if val_ema["accuracy"] >= val_live["accuracy"] else val_live
            chose = "ema" if val_ema["accuracy"] >= val_live["accuracy"] else "live"
            print(f"   [val ] live acc={val_live['accuracy']:.4f}  ema acc={val_ema['accuracy']:.4f}  -> using {chose}")
            print(f"   [val ] bal_acc={val_metrics['balanced_accuracy']:.4f}  f1_macro={val_metrics['f1_macro']:.4f}")

            row = {
                "epoch": epoch,
                **tr,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_bal_acc": val_metrics["balanced_accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "chose": chose,
            }
            self.history.append(row)

            self._save(epoch, val_metrics)
            with open(os.path.join(cfg.RESULTS_DIR, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        print(f"\n[done] best val acc = {self.best_val_acc:.4f}")
        return self.history