from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerModelConfig


class TransformerAutoencoder(nn.Module):
    """Transformer encoder + LSTM decoder autoencoder."""
    
    def __init__(self, cfg: TransformerModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        
        self.input_embedding = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, cfg.hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.latent_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input_embedding.weight)
            nn.init.zeros_(self.input_embedding.bias)
            nn.init.normal_(self.pos_encoding, std=0.02)
                
            for m in self.latent_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    if m.weight is not None:
                        m.weight.fill_(1.0)
                    if m.bias is not None:
                        m.bias.fill_(0.0)
            
            for name, param in self.decoder_lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)
                    h = param.size(0) // 4
                    param[h:2*h].fill_(1.0)
            
            nn.init.xavier_uniform_(self.decoder_fc.weight)
            nn.init.zeros_(self.decoder_fc.bias)
    
    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent vector [B, latent_dim]"""
        device = x.device
        lengths = lengths.to(device)
        B, S, _ = x.shape
        
        x_emb = self.input_embedding(x)
        pos_enc = self.pos_encoding[:, :S, :] if S <= self.pos_encoding.size(1) else F.pad(
            self.pos_encoding, (0, 0, 0, S - self.pos_encoding.size(1)), mode='constant', value=0
        )
        x_emb = x_emb + pos_enc
        
        mask = torch.arange(S, device=device).unsqueeze(0).expand(B, S) >= lengths.unsqueeze(1)
        x_encoded = self.encoder(x_emb, src_key_padding_mask=mask)
        
        valid_mask = (~mask).float().unsqueeze(-1)
        sum_pooled = (x_encoded * valid_mask).sum(dim=1)
        pooled = sum_pooled / lengths.float().unsqueeze(-1).clamp(min=1)
        
        z = self.latent_proj(pooled)
        return z
    
    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        z_rep = latent.unsqueeze(1).repeat(1, max_len, 1)
        dec_outputs, _ = self.decoder_lstm(z_rep)
        return self.decoder_fc(dec_outputs)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode"""
        latent = self.encode(x, lengths)
        recon = self.decode(latent, x.size(1))
        return recon, latent

