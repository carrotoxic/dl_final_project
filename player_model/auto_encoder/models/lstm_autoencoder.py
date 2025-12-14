from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig


class LSTMAutoencoder(nn.Module):
    """
    Temporal Autoencoder for Play-style Identification.
    
    Based on the approach: given trajectories X, encode each trajectory xi into
    a fixed-size latent representation zi, then decode back to reconstruct xi'.
    
    Architecture:
    1. Encoder: LSTM processes variable-length trajectories, each time-step
       feeds into its own LSTM cell, passing (ht, ct) sequentially.
       Final hidden state is projected to fixed-size latent vector zi.
    2. Decoder: LSTM reconstructs trajectory from latent vector zi.
    
    Formally: zi = Encoder(xi) and x'i = Decoder(zi)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder LSTM: processes each time-step sequentially
        # Each time-step feeds into its own LSTM cell, passing (ht, ct) forward
        # Non-stacked (num_layers=1) as per paper approach
        self.encoder_lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,  # Typically 1 for non-stacked approach
            batch_first=True,
        )

        # Projection to latent space: converts final LSTM hidden state to fixed-size zi
        # Input: [B, hidden_dim] -> Output: [B, latent_dim]
        self.fc_enc = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        # Decoder LSTM: reconstructs trajectory from latent vector
        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True
        )

        # Decoder: projects decoder LSTM output back to input dimension
        self.fc_dec = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform (LSTM + Linear)."""
        for name, param in self.named_parameters():
            # LSTM input weights
            if 'encoder_lstm' in name or 'decoder_lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # LSTM bias: zero init, then set forget gate bias to 1
                    param.data.fill_(0)
                    n = param.size(0)           # 4 * hidden_dim
                    start, end = n // 4, n // 2 # forget gate chunk
                    param.data[start:end].fill_(1)
            else:
                # Linear layers
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory xi into fixed-size latent representation zi.
        
        Process: Each time-step feeds into LSTM cell, passing (ht, ct) sequentially.
        Final hidden state is extracted and projected to latent space.
        
        Args:
            x: Input trajectories [B, T_max, input_dim]
            lengths: Actual lengths of each trajectory [B]
            
        Returns:
            z: Latent representations [B, latent_dim]
        """
        device = x.device
        lengths = lengths.to(device)
        B = x.shape[0]  # Batch size

        # 1) Packed sequence to handle variable-length trajectories
        # Each trajectory xi can have different length
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        # 2) LSTM processes sequence: each time-step t feeds into LSTM cell
        # Cells pass important information (ht, ct) forward through sequence
        outputs, (h_n, c_n) = self.encoder_lstm(packed)
        # outputs: [B, T_max, hidden_dim]
        # h_n: [num_layers, B, hidden_dim]
        # Contains final hidden state from the sequence

        # 3) Extract final hidden state from last layer
        # This represents the accumulated information from the entire sequence
        last_layer_h = h_n[-1]  # [B, hidden_dim] - last layer's hidden state

        # 4) Project to fixed-size latent space: zi = Encoder(xi)
        z = self.fc_enc(last_layer_h)  # [B, latent_dim]
        return z

    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Decode latent representation zi back to reconstructed trajectory x'i.
        
        Process: Expand latent vector along time dimension, then use LSTM decoder
        to reconstruct the original trajectory.
        
        Args:
            latent: Latent representations [B, latent_dim]
            max_len: Maximum sequence length for reconstruction
            
        Returns:
            recon: Reconstructed trajectories [B, T_max, input_dim]
        """
        B = latent.shape[0]

        # 1) Expand latent vector along time axis
        # Repeat zi for each time-step to feed into decoder LSTM
        z_rep = latent.unsqueeze(1).repeat(1, max_len, 1)

        # 2) Decoder LSTM reconstructs sequence from latent representation
        dec_outputs, _ = self.decoder_lstm(z_rep)           # [B, T, hidden_dim]

        # 3) Frame decoder: restore each frame (x, y) from decoder hidden states
        # Project decoder outputs back to original input dimension
        recon = self.fc_dec(dec_outputs)
        return recon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Encode trajectory and reconstruct it.
        
        Implements: zi = Encoder(xi) and x'i = Decoder(zi)
        
        Args:
            x: Input trajectories [B, T_max, input_dim]
            lengths: Actual lengths [B]
            
        Returns:
            recon: Reconstructed trajectories [B, T_max, input_dim]
            latent: Latent representations [B, latent_dim] (for clustering)
        """
        latent = self.encode(x, lengths)  # zi = Encoder(xi)
        recon = self.decode(latent, x.size(1))  # x'i = Decoder(zi)
        return recon, latent