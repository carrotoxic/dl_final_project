"""
Multi-Modal Fusion Autoencoder for combining trajectory latent features
with raw aggregated features.

Architecture:
1. Separate encoders for trajectory latent and raw features
2. Fusion layer to combine encoded representations
3. Joint latent space
4. Separate decoders to reconstruct both inputs

This allows the model to learn a joint representation that captures
relationships between trajectory patterns and game statistics.
"""

from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig


class FusionAutoencoder(nn.Module):
    """
    Multi-input autoencoder that fuses trajectory latent features with raw features.
    
    Inputs:
        - trajectory_latent: [B, trajectory_latent_dim] - from pre-trained LSTM autoencoder
        - raw_features: [B, raw_feature_dim] - game statistics (status, kills, coins, etc.)
    
    Architecture:
        trajectory_latent -> Encoder1 -> | 
                                          |-> Fusion -> Joint Latent -> Decoder1 -> recon_trajectory_latent
        raw_features      -> Encoder2 -> |                |                    |
                                         |                v                    |
                                         |         Decoder2 -> recon_raw_features
    """
    
    def __init__(
        self,
        trajectory_latent_dim: int = 8,
        raw_feature_dim: int = 10,  # status one-hot (3) + numeric features (7)
        joint_latent_dim: int = 16,  # Joint latent space dimension
        hidden_dim: int = 32,  # Hidden dimension for encoders/decoders
        fusion_hidden_dim: int = 32,  # Hidden dimension for fusion layer
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.trajectory_latent_dim = trajectory_latent_dim
        self.raw_feature_dim = raw_feature_dim
        self.joint_latent_dim = joint_latent_dim
        
        # Encoder for trajectory latent
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Encoder for raw features
        self.raw_encoder = nn.Sequential(
            nn.Linear(raw_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Fusion layer: combines encoded trajectory and raw features
        # Option 1: Concatenation + projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, joint_latent_dim),
        )
        
        # Alternative: Use attention-based fusion (commented out)
        # self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        # self.fusion_proj = nn.Linear(hidden_dim * 2, joint_latent_dim)
        
        # Decoder for trajectory latent
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(joint_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_latent_dim),
        )
        
        # Decoder for raw features
        self.raw_decoder = nn.Sequential(
            nn.Linear(joint_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, raw_feature_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(
        self,
        trajectory_latent: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode both inputs into joint latent representation.
        
        Args:
            trajectory_latent: [B, trajectory_latent_dim]
            raw_features: [B, raw_feature_dim]
        
        Returns:
            joint_latent: [B, joint_latent_dim]
        """
        # Encode both inputs separately
        traj_encoded = self.trajectory_encoder(trajectory_latent)  # [B, hidden_dim]
        raw_encoded = self.raw_encoder(raw_features)  # [B, hidden_dim]
        
        # Fuse: concatenate and project to joint latent space
        fused = torch.cat([traj_encoded, raw_encoded], dim=1)  # [B, hidden_dim * 2]
        joint_latent = self.fusion(fused)  # [B, joint_latent_dim]
        
        return joint_latent
    
    def decode(
        self,
        joint_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode joint latent to reconstruct both inputs.
        
        Args:
            joint_latent: [B, joint_latent_dim]
        
        Returns:
            recon_trajectory_latent: [B, trajectory_latent_dim]
            recon_raw_features: [B, raw_feature_dim]
        """
        recon_trajectory = self.trajectory_decoder(joint_latent)
        recon_raw = self.raw_decoder(joint_latent)
        
        return recon_trajectory, recon_raw
    
    def forward(
        self,
        trajectory_latent: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode both inputs, then decode to reconstruct.
        
        Args:
            trajectory_latent: [B, trajectory_latent_dim]
            raw_features: [B, raw_feature_dim]
        
        Returns:
            joint_latent: [B, joint_latent_dim]
            recon_trajectory_latent: [B, trajectory_latent_dim]
            recon_raw_features: [B, raw_feature_dim]
        """
        joint_latent = self.encode(trajectory_latent, raw_features)
        recon_trajectory, recon_raw = self.decode(joint_latent)
        
        return joint_latent, recon_trajectory, recon_raw


class AttentionFusionAutoencoder(nn.Module):
    """
    Alternative fusion autoencoder using attention mechanism for fusion.
    
    Uses cross-attention to learn how trajectory and raw features relate.
    """
    
    def __init__(
        self,
        trajectory_latent_dim: int = 8,
        raw_feature_dim: int = 10,
        joint_latent_dim: int = 16,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.trajectory_latent_dim = trajectory_latent_dim
        self.raw_feature_dim = raw_feature_dim
        self.joint_latent_dim = joint_latent_dim
        
        # Encoders
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.raw_encoder = nn.Sequential(
            nn.Linear(raw_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention-based fusion
        # Use trajectory as query, raw as key/value (or vice versa)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Project attended features to joint latent
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_latent_dim),
        )
        
        # Decoders (same as FusionAutoencoder)
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(joint_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trajectory_latent_dim),
        )
        
        self.raw_decoder = nn.Sequential(
            nn.Linear(joint_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, raw_feature_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(
        self,
        trajectory_latent: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode using attention-based fusion."""
        # Encode inputs
        traj_encoded = self.trajectory_encoder(trajectory_latent)  # [B, hidden_dim]
        raw_encoded = self.raw_encoder(raw_features)  # [B, hidden_dim]
        
        # Add sequence dimension for attention: [B, 1, hidden_dim]
        traj_seq = traj_encoded.unsqueeze(1)
        raw_seq = raw_encoded.unsqueeze(1)
        
        # Cross-attention: trajectory attends to raw features
        attended, _ = self.attention(traj_seq, raw_seq, raw_seq)  # [B, 1, hidden_dim]
        attended = attended.squeeze(1)  # [B, hidden_dim]
        
        # Combine attended and original
        fused = torch.cat([attended, traj_encoded], dim=1)  # [B, hidden_dim * 2]
        joint_latent = self.fusion_proj(fused)  # [B, joint_latent_dim]
        
        return joint_latent
    
    def decode(
        self,
        joint_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode joint latent."""
        recon_trajectory = self.trajectory_decoder(joint_latent)
        recon_raw = self.raw_decoder(joint_latent)
        return recon_trajectory, recon_raw
    
    def forward(
        self,
        trajectory_latent: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        joint_latent = self.encode(trajectory_latent, raw_features)
        recon_trajectory, recon_raw = self.decode(joint_latent)
        return joint_latent, recon_trajectory, recon_raw

