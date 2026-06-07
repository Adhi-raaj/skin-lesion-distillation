"""
Active Learning utilities.

Query strategies for selecting which unlabeled examples to label next.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class UncertaintySampler:
    """
    Base class for uncertainty-based active learning query strategies.
    """
    
    @staticmethod
    def entropy_sampling(model, unlabeled_loader, device, n_to_label):
        """
        Query examples with highest prediction entropy (most uncertain).
        
        Entropy = -Σ(p_i * log(p_i))
        - High entropy: model confused across classes → worth labeling
        - Low entropy: model confident → skip
        
        Args:
            model: PyTorch model in eval mode
            unlabeled_loader: DataLoader with (images, indices) for unlabeled data
            device: torch device
            n_to_label: number of examples to query
        
        Returns:
            indices: numpy array of indices to label (size: n_to_label)
        """
        uncertainties = []
        indices_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, indices) in enumerate(unlabeled_loader):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                # Entropy: -Σ p * log(p)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                uncertainties.extend(entropy.cpu().numpy())
                indices_list.extend(indices.numpy())
        
        uncertainties = np.array(uncertainties)
        indices_array = np.array(indices_list)
        
        # Return indices of top-k most uncertain examples
        top_k_idx = np.argsort(-uncertainties)[:n_to_label]
        
        return indices_array[top_k_idx]
    
    @staticmethod
    def margin_sampling(model, unlabeled_loader, device, n_to_label):
        """
        Query examples with smallest margin between top-2 predictions.
        
        Margin = max(p) - second_max(p)
        - Small margin: model uncertain between two classes
        - Large margin: model confident in top-1 class
        
        Args:
            model: PyTorch model in eval mode
            unlabeled_loader: DataLoader with (images, indices)
            device: torch device
            n_to_label: number of examples to query
        
        Returns:
            indices: numpy array of indices to label
        """
        margins = []
        indices_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, indices) in enumerate(unlabeled_loader):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                # Get top-2 probabilities
                top_2_probs = torch.topk(probs, k=2, dim=1)[0]
                margin = top_2_probs[:, 0] - top_2_probs[:, 1]
                
                margins.extend(margin.cpu().numpy())
                indices_list.extend(indices.numpy())
        
        margins = np.array(margins)
        indices_array = np.array(indices_list)
        
        # Return indices with smallest margin (most uncertain)
        bottom_k_idx = np.argsort(margins)[:n_to_label]
        
        return indices_array[bottom_k_idx]
    
    @staticmethod
    def least_confident(model, unlabeled_loader, device, n_to_label):
        """
        Query examples where max(p) is smallest (least confident predictions).
        
        Args:
            model: PyTorch model in eval mode
            unlabeled_loader: DataLoader with (images, indices)
            device: torch device
            n_to_label: number of examples to query
        
        Returns:
            indices: numpy array of indices to label
        """
        confidences = []
        indices_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, indices) in enumerate(unlabeled_loader):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                # Max probability = confidence
                max_probs = probs.max(dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
                indices_list.extend(indices.numpy())
        
        confidences = np.array(confidences)
        indices_array = np.array(indices_list)
        
        # Return indices with lowest confidence
        bottom_k_idx = np.argsort(confidences)[:n_to_label]
        
        return indices_array[bottom_k_idx]


def get_sampler(strategy_name="entropy"):
    """
    Factory function to get query strategy.
    
    Args:
        strategy_name: "entropy", "margin", or "least_confident"
    
    Returns:
        sampling function
    """
    strategies = {
        "entropy": UncertaintySampler.entropy_sampling,
        "margin": UncertaintySampler.margin_sampling,
        "least_confident": UncertaintySampler.least_confident,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Options: {list(strategies.keys())}")
    
    return strategies[strategy_name]
