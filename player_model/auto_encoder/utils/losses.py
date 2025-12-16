import torch
import torch.nn as nn


def masked_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor:
    """Masked MSE loss for variable-length sequences [B, T_max, D]"""
    device = recon.device
    B, T_max, D = recon.shape
    lengths = lengths.to(device)

    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)

    diff = (recon - target) ** 2
    diff = diff * mask

    per_sample_sum = diff.sum(dim=(1, 2))
    valid_elems = (lengths * D).clamp(min=1)
    per_sample_mse = per_sample_sum / valid_elems
    
    loss = (1/D) * per_sample_mse.mean()
    return loss


def fusion_reconstruction_loss(
    recon_trajectory: torch.Tensor,
    target_trajectory: torch.Tensor,
    recon_raw: torch.Tensor,
    target_raw: torch.Tensor,
    trajectory_weight: float = 1.0,
    raw_weight: float = 1.0,
) -> torch.Tensor:
    """Combined reconstruction loss for trajectory and raw features"""
    traj_loss = nn.functional.mse_loss(recon_trajectory, target_trajectory)
    raw_loss = nn.functional.mse_loss(recon_raw, target_raw)
    total_loss = trajectory_weight * traj_loss + raw_weight * raw_loss
    return total_loss, traj_loss, raw_loss