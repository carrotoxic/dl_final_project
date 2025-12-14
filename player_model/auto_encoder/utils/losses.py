"""Loss functions for auto-encoder training.

Implements the reconstruction loss: L_recon = 1/2 * ||(x - x')||²₂
where x is the original trajectory and x' is the reconstructed trajectory.
"""

import torch

def masked_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor:
    """Compute masked MSE loss for variable-length sequences.
    
    Implements: L_recon = 1/2 * ||(x - x')||²₂
    Only computes loss on valid timesteps (up to the actual length of each sequence),
    ignoring padded positions.
    
    Args:
        recon: Reconstructed trajectories x' of shape [B, T_max, D]
        target: Target trajectories x of shape [B, T_max, D]
        lengths: Actual lengths of each trajectory of shape [B]
        
    Returns:
        Mean reconstruction loss: 1/2 * mean(||(x - x')||²₂) over valid timesteps
    """
    device = recon.device
    B, T_max, D = recon.shape
    lengths = lengths.to(device)

    # Create mask for valid timesteps
    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)  # [B, T_max, 1]

    # Compute squared differences: ||(x - x')||²₂
    diff = (recon - target) ** 2  # [B, T_max, D]
    diff = diff * mask  # [B, T_max, D] - mask out padded positions

    # Sum over timesteps and dimensions for each sample
    per_sample_sum = diff.sum(dim=(1, 2))  # [B]

    # Normalize by number of valid elements per sample
    valid_elems = (lengths * D).clamp(min=1)  # [B]
    per_sample_mse = per_sample_sum / valid_elems  # [B]
    
    # Apply dimension factor: L_recon = 1/D * ||(x - x')||²₂
    loss = 1/D * per_sample_mse.mean()
    return loss

