from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerModelConfig


class TransformerAutoencoder(nn.Module):
    """Transformer-based sequence autoencoder.
    
    Architecture (as per paper):
    - Input embedding: Embed trajectory [B, S, F] into embedding vectors [B, S, D] with hidden dimension D
    - Positional encoding: Learnable positional encodings
    - Transformer encoder: Full self-attention mechanism (as in original Transformer)
      - Processes embedded sequence and outputs latent vectors [B, S, D]
    - Latent aggregation: Mean pooling over valid timesteps to get [B, D]
    - MLP decoder: Multi-layer perceptron that reconstructs trajectory [B, S, F] from latent vectors
    - Loss: L2 loss (MSE) between actual and reconstructed trajectories
    """
    def __init__(self, cfg: TransformerModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        
        # Input embedding: Embed trajectory [B, S, F] -> [B, S, D] where D is latent_dim
        self.input_embedding = nn.Linear(cfg.input_dim, cfg.latent_dim)
        
        # Learnable positional encoding
        self.max_seq_len = max_seq_len
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.max_seq_len, cfg.latent_dim))
        
        # Transformer encoder with full self-attention mechanism (as in original Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # MLP decoder: Multi-layer perceptron for trajectory reconstruction
        # Takes latent vectors and reconstructs original trajectory [B, S, F]
        decoder_hidden = cfg.ff_dim * 2
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(decoder_hidden, cfg.input_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.zeros_(self.input_embedding.bias)
        nn.init.normal_(self.pos_encoding, std=0.02)
    
    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode input sequences to per-timestep latent vectors.
        
        Process:
        1. Embed trajectory [B, S, F] into embedding vectors [B, S, D]
        2. Add positional encoding
        3. Pass through Transformer encoder with full self-attention
        
        Args:
            x: Input sequences [B, S, F] where F is number of features
            lengths: Actual sequence lengths [B]
            
        Returns:
            Per-timestep latent vectors [B, S, latent_dim]
        """
        B, S, feat_dim = x.shape
        device = x.device
        lengths = lengths.to(device)
        
        # Step 1: Embed trajectory into embedding vectors with dimension D (latent_dim)
        x_emb = self.input_embedding(x)  # [B, S, latent_dim]
        
        # Step 2: Add positional encoding
        if S > self.pos_encoding.size(1):
            pos_enc = F.pad(self.pos_encoding, (0, 0, 0, S - self.pos_encoding.size(1)), mode='constant', value=0)
        else:
            pos_enc = self.pos_encoding[:, :S, :]
        x_emb = x_emb + pos_enc
        
        # Create attention mask for padding positions
        max_len = x_emb.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)
        mask = mask >= lengths.unsqueeze(1)  # [B, S] - True for padding positions
        
        # Step 3: Transformer encoder with full self-attention mechanism
        # Uses full self-attention: each position can attend to all other positions
        # This captures long-term relationships (e.g., early trajectory points influence later ones)
        x_encoded = self.encoder(x_emb, src_key_padding_mask=mask)  # [B, S, latent_dim]
        
        return x_encoded
    
    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        """Decode latent representation to reconstructed sequences.
        
        Process:
        1. Expand latent vectors to sequence length
        2. Apply MLP decoder to reconstruct trajectory [B, S, F]
        
        Args:
            latent: Latent representation [B, latent_dim] from encoder
            max_len: Maximum sequence length S
            
        Returns:
            Reconstructed trajectories [B, S, F]
        """
        B = latent.shape[0]
        
        # Expand latent vectors to sequence length: [B, latent_dim] -> [B, S, latent_dim]
        z_rep = latent.unsqueeze(1).expand(B, max_len, self.cfg.latent_dim)  # [B, S, latent_dim]
        
        # Apply MLP decoder: Multi-layer perceptron for trajectory reconstruction
        # Non-autoregressive: each time step is reconstructed independently
        B, T, H = z_rep.shape
        z_flat = z_rep.reshape(B * T, H)
        recon_flat = self.decoder(z_flat)  # [B*T, F]
        recon = recon_flat.view(B, T, self.cfg.input_dim)  # [B, S, F]
        
        return recon
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode.
        
        Args:
            x: Input sequences [B, S, F]
            lengths: Actual sequence lengths [B]
            
        Returns:
            recon: Reconstructed sequences [B, S, F]
            latent: Latent representation [B, latent_dim] (pooled from per-timestep latents)
        """
        # Encode to per-timestep latents
        x_encoded = self.encode(x, lengths)  # [B, S, latent_dim]
        
        # Aggregate via mean pooling over valid timesteps for latent representation
        B, S, _ = x_encoded.shape
        device = x.device
        lengths = lengths.to(device)
        mask = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        mask = mask >= lengths.unsqueeze(1)  # [B, S] - True for padding positions
        
        valid_mask = (~mask).float().unsqueeze(-1)  # [B, S, 1]
        x_masked = x_encoded * valid_mask  # [B, S, latent_dim]
        sum_pooled = x_masked.sum(dim=1)  # [B, latent_dim]
        lengths_expanded = lengths.float().unsqueeze(-1).clamp(min=1)  # [B, 1]
        latent = sum_pooled / lengths_expanded  # [B, latent_dim]
        
        # Decode from pooled latent
        recon = self.decode(latent, x.size(1))
        return recon, latent

