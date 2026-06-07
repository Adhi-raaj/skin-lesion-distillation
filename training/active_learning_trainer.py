"""
Active Learning Trainer with per-cycle checkpointing.

Iteratively trains on expanding labeled dataset using uncertainty-based query strategy.
Each cycle: Train → Eval → Query uncertain examples → Expand labeled set

Checkpoint structure:
- Saves model after each cycle
- Auto-resumes from latest cycle on crash
- Tracks labeled/unlabeled split per cycle
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils.dataset import HAMDataset, get_train_transforms, get_val_transforms
from utils.metrics import compute_metrics
from models import build_model, ModelEMA
from models.active_learning import get_sampler
from training.trainer import Trainer


class ActiveLearningTrainer:
    """
    Active Learning trainer that iteratively queries uncertain examples.
    
    Workflow:
    1. Cycle 0: Train on 20% labeled (random selection)
    2. Cycle 1: Query 10% most uncertain → Train on 30% labeled
    3. Cycle 2: Query 10% more → Train on 40% labeled
    ... repeat until reaching 100% or desired cycles
    
    Features:
    - Per-cycle checkpointing (resume on crash)
    - Tracks labeled/unlabeled indices
    - Uncertainty-based query strategies (entropy, margin, least_confident)
    - Compares vs. random sampling baseline
    """
    
    def __init__(self, cfg, model_checkpoint_path):
        """
        Args:
            cfg: Config object
            model_checkpoint_path: path to pre-trained model checkpoint (for uncertainty queries)
        """
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.results_dir = cfg.AL_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"[al] Loading query model from: {model_checkpoint_path}")

        # Load pre-trained model for uncertainty estimation
        self.query_model = build_model(cfg).to(self.device)

        ckpt = torch.load(
            model_checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Handle different checkpoint formats safely
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.query_model.load_state_dict(ckpt["model"], strict=False)
        else:
            self.query_model.load_state_dict(ckpt, strict=False)

        self.query_model.eval()

        for param in self.query_model.parameters():
            param.requires_grad = False
        
        print(f"[al] Query model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # ---- Load train/val/test data ----
        train_csv = os.path.join(cfg.PROCESSED_DIR, "train.csv")
        val_csv = os.path.join(cfg.PROCESSED_DIR, "val.csv")
        test_csv = os.path.join(cfg.PROCESSED_DIR, "test.csv")
        
        assert os.path.isfile(train_csv), f"missing {train_csv}. Run prepare_dataset.py first."
        
        self.train_ds = HAMDataset(train_csv, transform=get_train_transforms(cfg.IMG_SIZE))
        self.val_ds = HAMDataset(val_csv, transform=get_val_transforms(cfg.IMG_SIZE))
        self.test_ds = HAMDataset(test_csv, transform=get_val_transforms(cfg.IMG_SIZE))
        
        total_train = len(self.train_ds)
        print(f"[al] Total training samples: {total_train}")
        
        # ---- Initialize labeled/unlabeled split ----
        self.all_indices = np.arange(total_train)
        self.labeled_indices = None
        self.unlabeled_indices = None
        self.current_cycle = 0
        self.history = []
        self.random_history = []  # For baseline comparison
        
        # Try to resume from previous AL run
        self._try_resume()
        
        # If not resuming, start fresh
        if self.labeled_indices is None:
            self._initialize_labeled_set()
    
    # --------------------------------------------------------------------- #
    #  checkpoint I/O for Active Learning                                   #
    # --------------------------------------------------------------------- #
    
    def _al_ckpt_path(self, cycle):
        """Path to AL checkpoint for specific cycle"""
        return os.path.join(self.results_dir, f"al_cycle_{cycle:03d}.pth")
    
    def _al_state_path(self):
        """Path to AL state (labeled/unlabeled indices)"""
        return os.path.join(self.results_dir, f"al_state.json")
    
    def _latest_al_ckpt(self):
        """Find latest AL checkpoint"""
        files = sorted(glob.glob(os.path.join(self.results_dir, "al_cycle_*.pth")))
        return files[-1] if files else None
    
    def _save_al_state(self):
        """Save labeled/unlabeled indices and cycle info"""
        state = {
            "cycle": self.current_cycle,
            "labeled_indices": self.labeled_indices.tolist(),
            "unlabeled_indices": self.unlabeled_indices.tolist(),
            "labeled_ratio": len(self.labeled_indices) / len(self.all_indices),
            "history": self.history,
            "random_history": self.random_history,
        }
        
        with open(self._al_state_path(), "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"      [save] AL state saved: cycle {self.current_cycle}, "
              f"labeled {len(self.labeled_indices)}/{len(self.all_indices)}")
    
    def _try_resume(self):
        """Resume from latest AL checkpoint if exists"""
        state_path = self._al_state_path()
        
        if not os.path.isfile(state_path):
            print("[al-resume] No previous AL run found, starting fresh")
            return
        
        print(f"[al-resume] Loading AL state from: {state_path}")
        with open(state_path, "r") as f:
            state = json.load(f)
        
        self.current_cycle = state["cycle"]
        self.labeled_indices = np.array(state["labeled_indices"])
        self.unlabeled_indices = np.array(state["unlabeled_indices"])
        self.history = state["history"]
        self.random_history = state.get("random_history", [])
        
        print(f"[al-resume] Resuming at cycle {self.current_cycle}")
        print(f"[al-resume] Labeled: {len(self.labeled_indices)}, "
              f"Unlabeled: {len(self.unlabeled_indices)}")
    
    def _initialize_labeled_set(self):
        """Initialize with random subset of data"""
        total = len(self.all_indices)
        n_labeled = int(total * self.cfg.AL_INITIAL_RATIO)
        
        np.random.seed(self.cfg.SEED)
        labeled_idx = np.random.choice(self.all_indices, size=n_labeled, replace=False)
        
        self.labeled_indices = np.sort(labeled_idx)
        self.unlabeled_indices = np.array([i for i in self.all_indices if i not in self.labeled_indices])
        
        print(f"[al] Initialized labeled set: {len(self.labeled_indices)}/{total} "
              f"({len(self.labeled_indices)/total*100:.1f}%)")
        
        self._save_al_state()
    
    # --------------------------------------------------------------------- #
    #  uncertainty query & data sampling                                    #
    # --------------------------------------------------------------------- #
    
    def _get_labeled_loader(self, batch_size=None):
        """DataLoader for labeled subset"""
        batch_size = batch_size or self.cfg.AL_BATCH_SIZE
        subset = Subset(self.train_ds, self.labeled_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, 
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def _get_test_loader(self, batch_size=None):
        """DataLoader for test set"""
        batch_size = batch_size or self.cfg.AL_BATCH_SIZE
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def _get_val_loader(self, batch_size=None):
        """DataLoader for val set"""
        batch_size = batch_size or self.cfg.AL_BATCH_SIZE
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def _get_unlabeled_loader(self, batch_size=None):
        """DataLoader for unlabeled subset (for uncertainty estimation)"""
        batch_size = batch_size or self.cfg.AL_BATCH_SIZE
        subset = Subset(self.train_ds, self.unlabeled_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def _query_uncertain_examples(self, n_to_label):
        """
        Query n_to_label most uncertain examples using active learning strategy.
        
        Returns:
            numpy array of indices (in the unlabeled pool) to label next
        """
        print(f"\n   [query] Running {self.cfg.AL_UNCERTAINTY_METHOD} sampling...")
        
        sampler_fn = get_sampler(self.cfg.AL_UNCERTAINTY_METHOD)
        
        unlabeled_loader = self._get_unlabeled_loader()
        
        # Query function needs indices relative to unlabeled pool
        queried_pool_indices = sampler_fn(
            self.query_model,
            unlabeled_loader,
            self.device,
            n_to_label
        )
        
        # Convert back to absolute indices
        queried_absolute = self.unlabeled_indices[queried_pool_indices]
        
        print(f"   [query] Selected {len(queried_absolute)} examples to label")
        
        return queried_absolute
    
    def _generate_random_baseline(self, n_to_label):
        """Generate random baseline for comparison"""
        if len(self.unlabeled_indices) == 0:
            return np.array([])
        
        random_indices = np.random.choice(
            self.unlabeled_indices,
            size=min(n_to_label, len(self.unlabeled_indices)),
            replace=False
        )
        
        return random_indices
    
    # --------------------------------------------------------------------- #
    #  training single cycle                                                #
    # --------------------------------------------------------------------- #
    
    def _train_cycle(self, cycle):
        """
        Train model on current labeled subset for one cycle.
        
        Returns:
            dict with metrics
        """
        print(f"\n[cycle {cycle}] Training on labeled subset...")
        print(f"[cycle {cycle}] Labeled examples: {len(self.labeled_indices)}")
        
        # Create fresh trainer for this cycle
        labeled_loader = self._get_labeled_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        
        # Train a new model from scratch on labeled data
        cfg_copy = self.cfg
        trainer = Trainer(cfg_copy)
        
        # Override dataloaders to use only labeled subset
        trainer.train_loader = labeled_loader
        trainer.val_loader = val_loader
        
        # Fit for some epochs (can be shorter than full training)
        history = trainer.fit()
        
        # Evaluate on test set
        test_metrics = trainer._evaluate(test_loader, trainer.ema.module)
        
        results = {
            "cycle": cycle,
            "labeled_count": len(self.labeled_indices),
            "labeled_ratio": len(self.labeled_indices) / len(self.all_indices),
            "test_accuracy": test_metrics["accuracy"],
            "test_bal_acc": test_metrics["balanced_accuracy"],
            "test_f1": test_metrics["f1_macro"],
            "train_loss": history[-1]["train_loss"] if history else 0.0,
        }
        
        return results, trainer
    
    # --------------------------------------------------------------------- #
    #  main active learning loop                                            #
    # --------------------------------------------------------------------- #
    
    def run_active_learning_cycles(self):
        """
        Main AL loop: iteratively select examples and train.
        
        Returns:
            DataFrame with results
        """
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING: {self.cfg.AL_N_CYCLES} cycles")
        print(f"{'='*70}")
        print(f"Strategy: {self.cfg.AL_UNCERTAINTY_METHOD}")
        print(f"Initial ratio: {self.cfg.AL_INITIAL_RATIO:.1%}")
        print(f"Query ratio per cycle: {self.cfg.AL_QUERY_RATIO:.1%}")
        
        total_train = len(self.all_indices)
        query_size = int(total_train * self.cfg.AL_QUERY_RATIO)
        
        # Start from current_cycle (in case of resume)
        for cycle in range(self.current_cycle, self.cfg.AL_N_CYCLES):
            print(f"\n{'='*70}")
            print(f"CYCLE {cycle + 1}/{self.cfg.AL_N_CYCLES}")
            print(f"{'='*70}")
            
            labeled_ratio = len(self.labeled_indices) / total_train
            print(f"Current labeled: {len(self.labeled_indices)}/{total_train} ({labeled_ratio:.1%})")
            print(f"Current unlabeled: {len(self.unlabeled_indices)}/{total_train}")
            
            # Train on current labeled set
            results, trainer = self._train_cycle(cycle)
            self.history.append(results)
            
            print(f"\n[cycle {cycle}] Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"[cycle {cycle}] Test Bal. Acc: {results['test_bal_acc']:.4f}")
            
            # Save checkpoint after training
            ckpt_path = self._al_ckpt_path(cycle)
            torch.save(trainer.model.state_dict(), ckpt_path)
            print(f"[cycle {cycle}] Checkpoint saved: {ckpt_path}")
            
            # Query next batch if not last cycle
            if cycle < self.cfg.AL_N_CYCLES - 1 and len(self.unlabeled_indices) > 0:
                print(f"\n[cycle {cycle}] Querying next batch...")
                
                # Active Learning query
                queried_al = self._query_uncertain_examples(query_size)
                
                # Random baseline query (for comparison)
                queried_random = self._generate_random_baseline(query_size)
                
                # Add to labeled set
                self.labeled_indices = np.sort(np.concatenate([self.labeled_indices, queried_al]))
                self.unlabeled_indices = np.array([i for i in self.all_indices 
                                                   if i not in self.labeled_indices])
                
                # For baseline: track what random would have selected
                self.random_history.append({
                    "cycle": cycle,
                    "would_select": queried_random.tolist(),
                })
                
                self.current_cycle = cycle + 1
                self._save_al_state()
            else:
                print(f"[cycle {cycle}] Final cycle or no unlabeled data left")
                self.current_cycle = cycle + 1
                self._save_al_state()
        
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING COMPLETE")
        print(f"{'='*70}")
        
        # Return results as DataFrame
        results_df = pd.DataFrame(self.history)
        return results_df