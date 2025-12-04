"""Loss functions for auto-encoder training."""

import torch


def masked_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute masked MSE loss for variable-length sequences.
    
    Only computes loss on valid timesteps (up to the actual length of each sequence),
    ignoring padded positions.
    
    Args:
        recon: Reconstructed trajectories of shape [B, T_max, D]
        target: Target trajectories of shape [B, T_max, D]
        lengths: Actual lengths of each trajectory of shape [B]
        
    Returns:
        Mean MSE loss across all valid timesteps
    """
    device = recon.device
    B, T_max, D = recon.shape
    lengths = lengths.to(device)

    # Create mask for valid timesteps
    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)

    # Compute squared differences and apply mask
    diff = (recon - target) ** 2
    diff = diff * mask

    # Sum over timesteps and dimensions for each sample
    per_sample_sum = diff.sum(dim=(1, 2))

    # Normalize by number of valid elements per sample
    valid_elems = (lengths * D).clamp(min=1)
    per_sample_mse = per_sample_sum / valid_elems
    loss = per_sample_mse.mean()
    return loss

