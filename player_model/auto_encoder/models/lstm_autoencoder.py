from typing import Tuple

import torch
import torch.nn as nn

from ..config import LSTMModelConfig


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder: zi = Encoder(xi), x'i = Decoder(zi)"""
    def __init__(self, cfg: LSTMModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder_lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        self.fc_dec = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self._init_weights()
    

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if "weight_ih" in name:
                            nn.init.xavier_uniform_(param)
                        elif "weight_hh" in name:
                            nn.init.orthogonal_(param)
                        elif "bias" in name:
                            param.fill_(0.0)
                            h = param.size(0) // 4
                            param[h:2*h].fill_(1.0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                    m.weight is not None and m.weight.fill_(1.0)
                    m.bias is not None and m.bias.fill_(0.0)


    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent vector [B, latent_dim]"""
        device = x.device
        lengths = lengths.to(device)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        _, (h_n, _) = self.encoder_lstm(packed)
        last_layer_h = h_n[-1]
        z = self.fc_enc(last_layer_h)
        return z

    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        """Decode latent vector to reconstructed trajectory [B, T_max, input_dim]"""
        z_rep = latent.unsqueeze(1).repeat(1, max_len, 1)
        dec_outputs, _ = self.decoder_lstm(z_rep)
        recon = self.fc_dec(dec_outputs)
        return recon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode"""
        latent = self.encode(x, lengths)
        recon = self.decode(latent, x.size(1))
        return recon, latent