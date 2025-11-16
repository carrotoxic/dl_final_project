from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig


class LSTMAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder_lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

        self.to_latent = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self.to_dec_h = nn.Linear(cfg.latent_dim, cfg.hidden_dim * cfg.num_layers)
        self.to_dec_c = nn.Linear(cfg.latent_dim, cfg.hidden_dim * cfg.num_layers)

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder_lstm(x)
        device = outputs.device
        lengths = lengths.to(device)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        last_outputs = outputs.gather(1, idx).squeeze(1)

        latent = self.to_latent(last_outputs)
        return latent

    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        B = latent.size(0)
        device = latent.device

        L = self.cfg.num_layers
        H = self.cfg.hidden_dim

        h0 = self.to_dec_h(latent).view(B, L, H).permute(1, 0, 2).contiguous()
        c0 = self.to_dec_c(latent).view(B, L, H).permute(1, 0, 2).contiguous()

        dec_inputs = torch.zeros(B, max_len, self.cfg.input_dim, device=device)
        dec_outputs, _ = self.decoder_lstm(dec_inputs, (h0, c0))
        recon = self.output_proj(dec_outputs)
        return recon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, lengths)
        recon = self.decode(latent, x.size(1))
        return recon, latent