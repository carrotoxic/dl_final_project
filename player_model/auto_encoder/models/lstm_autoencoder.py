from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig


class LSTMAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.frame_encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.frame_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.frame_hidden_dim, cfg.frame_hidden_dim),
            nn.ReLU(),
        )

        self.encoder_lstm = nn.LSTM(
            input_size=cfg.frame_hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )


        self.to_latent = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.latent_dim),
        )


        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.frame_decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.frame_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.frame_hidden_dim, cfg.input_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = x.device
        lengths = lengths.to(device)

        B, T, D = x.shape

        # 1) Frame encoder: each frame (x, y) -> frame_hidden_dim
        x_flat = x.reshape(B * T, D)                      # [B*T, input_dim]
        frame_emb_flat = self.frame_encoder(x_flat)       # [B*T, frame_hidden_dim]
        frame_emb = frame_emb_flat.view(B, T, -1)         # [B, T, frame_hidden_dim]

        # 2) Packed sequence to handle variable length
        packed = nn.utils.rnn.pack_padded_sequence(
            frame_emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, c_n) = self.encoder_lstm(packed)
        # h_n: [num_layers * num_directions, B, hidden_dim]

        # Get the hidden state of the last layer of the BiLSTM
        last_layer_h = h_n.view(self.cfg.num_layers,
                                2,
                                B,
                                self.cfg.hidden_dim)[-1]  # [2, B, hidden_dim]
        last_layer_h = last_layer_h.permute(1, 0, 2).contiguous()  # [B, num_directions, hidden_dim]
        last_layer_h = last_layer_h.view(B, -1)                    # [B, hidden_dim * num_directions]

        # 3) latent z
        z = self.to_latent(last_layer_h)                           # [B, latent_dim]
        return z

    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        B, Lz = latent.shape
        device = latent.device

        # 1) Repeat z along the time axis: [B, 1, Lz] -> [B, T, Lz]
        z_rep = latent.unsqueeze(1).expand(B, max_len, Lz)  # [B, T, latent_dim]

        # 2) decoder BiLSTM (initial state is 0)
        dec_outputs, _ = self.decoder_lstm(z_rep)           # [B, T, hidden_dim * num_directions]

        # 3) frame decoder to restore each frame (x, y)
        B, T, H2 = dec_outputs.shape
        dec_flat = dec_outputs.reshape(B * T, H2)           # [B*T, H2]
        x_hat_flat = self.frame_decoder(dec_flat)           # [B*T, input_dim]
        recon = x_hat_flat.view(B, T, self.cfg.input_dim)   # [B, T, input_dim]

        return recon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, lengths)
        recon = self.decode(latent, x.size(1))
        return recon, latent