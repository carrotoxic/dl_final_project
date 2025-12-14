"""Metrics computation and evaluation functions for auto-encoder."""

from typing import Dict

import torch


def compute_euclidean_distance_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute Euclidean distance metrics in original (denormalized) space.
    
    Following the approach in arXiv papers: since computing loss on normalized data
    "hides the actual distances", we denormalize both original and reconstructed
    trajectories back to their original coordinate space before computing the mean
    Euclidean distance per point. This gives interpretable metrics in pixel space.
    
    Args:
        recon: Reconstructed trajectories [B, T_max, D] in normalized space [0, 1]
        target: Target trajectories [B, T_max, D] in normalized space [0, 1]
        lengths: Actual lengths of each trajectory [B]
        means: Per-trajectory normalization means [B, D]
        stds: Per-trajectory normalization stds [B, D]
        device: Device to run computation on
        
    Returns:
        Dictionary with mean Euclidean distance per point in original pixel space
    """
    B, T_max, D = recon.shape
    lengths = lengths.to(device)
    means = means.to(device)
    stds = stds.to(device)
    
    # Denormalize both reconstruction and target back to original space
    # This reveals the actual pixel distances that were hidden by normalization
    recon_orig = recon * stds.unsqueeze(1) + means.unsqueeze(1)  # [B, T_max, D]
    target_orig = target * stds.unsqueeze(1) + means.unsqueeze(1)  # [B, T_max, D]
    
    # Create mask for valid timesteps
    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)  # [B, T_max, 1]
    
    # Compute Euclidean distance per point
    euclidean_dist = torch.sqrt(((recon_orig - target_orig) ** 2).sum(dim=-1))  # [B, T_max]
    euclidean_dist = euclidean_dist * mask.squeeze(-1)  # [B, T_max]
    
    # Sum over timesteps for each sample, then divide by length
    per_sample_dist_sum = euclidean_dist.sum(dim=1)  # [B]
    per_sample_mean_dist = per_sample_dist_sum / lengths.float()  # [B]
    
    # Return mean across all samples and all points
    mean_euclidean_dist = per_sample_mean_dist.mean().item()
    
    return {
        "mean_euclidean_distance": mean_euclidean_dist,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_original_metrics: bool = True,
    masked_mse_loss_fn = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Computes normalized loss and optionally original space metrics (Euclidean distance).
    
    Args:
        model: The autoencoder model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        compute_original_metrics: Whether to compute metrics in original space
        masked_mse_loss_fn: Loss function to use. If None, will import from losses module.
        
    Returns:
        Dictionary with 'normalized_loss' and optionally 'mean_euclidean_distance'
    """
    if masked_mse_loss_fn is None:
        from .losses import masked_mse_loss
        masked_mse_loss_fn = masked_mse_loss
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_euclidean_dists = []
    
    with torch.no_grad():
        for batch in dataloader:
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)
            means = batch["normalization_means"].to(device)
            stds = batch["normalization_stds"].to(device)
            
            recon, _ = model(trajectories, lengths)
            
            # Normalized loss (for training)
            loss = masked_mse_loss_fn(recon, trajectories, lengths)
            total_loss += loss.item()
            
            # Original space metrics (for reporting)
            if compute_original_metrics:
                metrics = compute_euclidean_distance_metrics(
                    recon, trajectories, lengths, means, stds, device
                )
                all_euclidean_dists.append(metrics["mean_euclidean_distance"])
            
            num_batches += 1
    
    results = {
        "normalized_loss": total_loss / num_batches if num_batches > 0 else 0.0,
    }
    
    if compute_original_metrics and all_euclidean_dists:
        results["mean_euclidean_distance"] = sum(all_euclidean_dists) / len(all_euclidean_dists)
    
    return results


def compute_training_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute training metrics in original space.
    
    Args:
        model: The autoencoder model
        dataloader: DataLoader for the training dataset
        device: Device to run computation on
        
    Returns:
        Mean Euclidean distance in original pixel space
    """
    # Save current training state
    was_training = model.training
    model.eval()
    euclidean_dists = []
    
    with torch.no_grad():
        for batch in dataloader:
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)
            means = batch["normalization_means"].to(device)
            stds = batch["normalization_stds"].to(device)
            
            recon, _ = model(trajectories, lengths)
            metrics = compute_euclidean_distance_metrics(
                recon, trajectories, lengths, means, stds, device
            )
            euclidean_dists.append(metrics["mean_euclidean_distance"])
    
    # Restore training state
    if was_training:
        model.train()
    
    return sum(euclidean_dists) / len(euclidean_dists) if euclidean_dists else 0.0

