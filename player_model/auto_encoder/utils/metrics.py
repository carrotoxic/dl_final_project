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
    """Compute Euclidean distance in original (denormalized) space"""
    B, T_max, D = recon.shape
    lengths = lengths.to(device)
    means = means.to(device)
    stds = stds.to(device)
    
    recon_orig = recon * stds.unsqueeze(1) + means.unsqueeze(1)
    target_orig = target * stds.unsqueeze(1) + means.unsqueeze(1)
    
    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)
    
    euclidean_dist = torch.sqrt(((recon_orig - target_orig) ** 2).sum(dim=-1))
    euclidean_dist = euclidean_dist * mask.squeeze(-1)
    
    per_sample_dist_sum = euclidean_dist.sum(dim=1)
    per_sample_mean_dist = per_sample_dist_sum / lengths.float()
    mean_euclidean_dist = per_sample_mean_dist.mean().item()
    
    return {"mean_euclidean_distance": mean_euclidean_dist}


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_original_metrics: bool = True,
    masked_mse_loss_fn = None,
) -> Dict[str, float]:
    """Evaluate model on dataset"""
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
            loss = masked_mse_loss_fn(recon, trajectories, lengths)
            total_loss += loss.item()
            
            if compute_original_metrics:
                metrics = compute_euclidean_distance_metrics(
                    recon, trajectories, lengths, means, stds, device
                )
                all_euclidean_dists.append(metrics["mean_euclidean_distance"])
            
            num_batches += 1
    
    results = {"normalized_loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    if compute_original_metrics and all_euclidean_dists:
        results["mean_euclidean_distance"] = sum(all_euclidean_dists) / len(all_euclidean_dists)
    
    return results


def compute_training_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute training metrics in original space"""
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
    
    if was_training:
        model.train()
    
    return sum(euclidean_dists) / len(euclidean_dists) if euclidean_dists else 0.0

