"""
Knowledge Distillation Loss and utilities.

Teacher generates soft targets (probability distributions).
Student learns to mimic these soft targets + hard labels.

Loss = (1 - alpha) * CE(student, labels) + alpha * KL(student_soft, teacher_soft)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - Cross-entropy loss on hard labels (ground truth)
    - KL divergence on soft targets (teacher guidance)
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature (float): Temperature for softening probabilities.
                                Higher = softer targets (more uniform).
                                Typical: 3-5
            alpha (float): Weight of distillation loss (KL).
                          alpha=0.7 means: 70% distill, 30% hard CE loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (B, num_classes) - raw model outputs from student
            teacher_logits: (B, num_classes) - raw model outputs from teacher (detached)
            labels: (B,) - ground truth class indices
        
        Returns:
            loss: scalar tensor
        """
        # Hard target loss: standard cross-entropy on ground truth
        ce = self.ce_loss(student_logits, labels)
        
        # Soft target loss: KL divergence between student and teacher distributions
        # Temperature scaling makes distributions softer (less peaked)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL(teacher || student) with temperature scaling factor
        kl = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combine: weighted average
        total_loss = (1 - self.alpha) * ce + self.alpha * kl
        
        return total_loss


class DistillationMetrics:
    """Track distillation-specific metrics during training"""
    
    @staticmethod
    def compute_logit_similarity(student_logits, teacher_logits):
        """
        Measure how similar student and teacher logits are.
        Useful for debugging: should increase as training progresses.
        
        Returns: cosine similarity (0-1, higher = more similar)
        """
        student_norm = F.normalize(student_logits, p=2, dim=1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=1)
        
        similarity = (student_norm * teacher_norm).sum(dim=1).mean()
        return similarity.item()
    
    @staticmethod
    def compute_agreement(student_logits, teacher_logits):
        """
        Measure how often student and teacher predict the same class.
        
        Returns: agreement rate (0-1, higher = more aligned)
        """
        student_pred = student_logits.argmax(dim=1)
        teacher_pred = teacher_logits.argmax(dim=1)
        
        agreement = (student_pred == teacher_pred).float().mean()
        return agreement.item()