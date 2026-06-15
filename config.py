"""
Central configuration for the rare disease (HAM10000) classification pipeline.
All hyperparameters live here so you can tune in one place.
Tuned for RTX 3060 12GB.

NEW: Added Knowledge Distillation & Active Learning configs
"""
import os
import torch

class Config:
    # ---------- Paths ----------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "HAM10000")        # raw dataset
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")
    DISTILL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "distill_T2")
    AL_DIR = os.path.join(PROJECT_ROOT, "results", "active_learning")

    # ---------- Dataset ----------
    NUM_CLASSES = 7
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    # HAM10000 metadata CSV name (default from the official ISIC release)
    METADATA_CSV = "HAM10000_metadata.csv"
    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

    # ---------- Splits ----------
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    SEED = 42

    # ---------- Model ----------
    # Options: 'efficientnet_b3', 'efficientnet_b4', 'convnext_tiny', 'convnext_small'
    # b3 is the sweet spot for 12GB; b4 works with smaller batch.
    BACKBONE = "mobilenetv2_100"
    PRETRAINED = True
    DROPOUT = 0.4

    # ---------- Training ----------
    IMG_SIZE = 300              # b3 native; lower to 224 for faster runs
    BATCH_SIZE = 32             # safe for 12GB at 300px with b3 + AMP
    NUM_WORKERS = 4             # Windows: keep <= 4 to avoid spawn issues
    EPOCHS = 20
    WARMUP_EPOCHS = 3
    BASE_LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.0
    GRAD_ACCUM_STEPS = 1        # bump to 2 if you OOM at batch 32
    GRAD_CLIP = 1.0
    USE_AMP = True              # mixed precision — big VRAM saving

    # ---------- Loss / sampling for class imbalance ----------
    USE_FOCAL_LOSS = False
    FOCAL_GAMMA = 2.0
    USE_CLASS_WEIGHTS = False
    USE_WEIGHTED_SAMPLER = False

    # ---------- Augmentation ----------
    USE_MIXUP = False
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5            # prob of applying mixup OR cutmix per batch

    # ---------- Checkpointing ----------
    SAVE_EVERY_EPOCH = True     # write a fresh checkpoint each epoch
    KEEP_LAST_N = 3             # only keep the last N epoch checkpoints
    SAVE_BEST = True            # always keep best_model.pth
    RESUME = False               # auto-resume from latest if it exists

    # ---------- Evaluation ----------
    USE_TTA = True              # test-time augmentation (5-crop + flip)
    TTA_STEPS = 8

    # ========== NEW: KNOWLEDGE DISTILLATION CONFIG ==========
    # Knowledge Distillation settings
    USE_DISTILLATION = False    # Set True to enable distillation training
    DISTILL_TEMPERATURE = 2.0   # Higher = softer targets
    DISTILL_ALPHA = 0.7         # Weight of KL loss (0.7 = 70% distill, 30% CE)
    TEACHER_BACKBONE = "efficientnet_b3"  # Teacher model backbone
    STUDENT_BACKBONE = "mobilenetv2_100"  # ← CHANGED (was mobilenet_v3_small)

    # Student-specific training
    DISTILL_BATCH_SIZE = 32     # Can be different from teacher
    DISTILL_EPOCHS = 80
    DISTILL_BASE_LR = 2e-4      # Students learn slower
    
    # ========== NEW: ACTIVE LEARNING CONFIG ==========
    USE_ACTIVE_LEARNING = True 
    AL_INITIAL_RATIO = 0.2        # Start with 20% labeled data
    AL_QUERY_RATIO = 0.1          # Query 10% more each cycle
    AL_N_CYCLES = 5               # Number of AL cycles
    AL_UNCERTAINTY_METHOD = "entropy"  # Options: entropy, margin, bald
    AL_BATCH_SIZE = 32

    # ---------- Hardware ----------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True
    BENCHMARK = True            # cudnn.benchmark for speed (fixed input sizes)

    @classmethod
    def show(cls):
        print("=" * 60)
        print("CONFIG")
        print("=" * 60)
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and not callable(v):
                print(f"  {k:24s} : {v}")
        print("=" * 60)


def ensure_dirs():
    for d in [Config.PROCESSED_DIR, Config.CHECKPOINT_DIR,
              Config.RESULTS_DIR, Config.LOG_DIR, Config.DISTILL_DIR, Config.AL_DIR]:
        os.makedirs(d, exist_ok=True)