from typing import Tuple

import torch
import torch.nn as nn

from ..config import FusionModelConfig


class TrajectoryFeatureFusionAutoencoder(nn.Module):
    """Fusion autoencoder: fuses trajectory sequences and aggregated features"""
    
    def __init__(self, cfg: FusionModelConfig):
        super().__init__()
        
        self.trajectory_input_dim = cfg.trajectory_input_dim
        self.raw_feature_dim = cfg.raw_feature_dim
        self.joint_latent_dim = cfg.joint_latent_dim
        self.trajectory_hidden_dim = cfg.trajectory_hidden_dim
        
        self.trajectory_encoder_lstm = nn.LSTM(
            input_size=cfg.trajectory_input_dim,
            hidden_size=cfg.trajectory_hidden_dim,
            num_layers=cfg.num_lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_lstm_layers > 1 else 0,
        )
        
        self.trajectory_encoder_proj = nn.Sequential(
            nn.Linear(cfg.trajectory_hidden_dim, cfg.trajectory_hidden_dim),
            nn.BatchNorm1d(cfg.trajectory_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        
        self.features_encoder = nn.Sequential(
            nn.Linear(cfg.raw_feature_dim, cfg.feature_hidden_dim),
            nn.BatchNorm1d(cfg.feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.feature_hidden_dim, cfg.feature_hidden_dim),
            nn.BatchNorm1d(cfg.feature_hidden_dim),
            nn.ReLU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(cfg.trajectory_hidden_dim + cfg.feature_hidden_dim, cfg.fusion_hidden_dim),
            nn.BatchNorm1d(cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.joint_latent_dim),
        )
        
        self.trajectory_decoder_lstm = nn.LSTM(
            input_size=cfg.joint_latent_dim,
            hidden_size=cfg.trajectory_hidden_dim,
            num_layers=cfg.num_lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_lstm_layers > 1 else 0,
        )
        
        self.trajectory_decoder_proj = nn.Linear(cfg.trajectory_hidden_dim, cfg.trajectory_input_dim)
        
        self.features_decoder = nn.Sequential(
            nn.Linear(cfg.joint_latent_dim, cfg.feature_hidden_dim),
            nn.BatchNorm1d(cfg.feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.feature_hidden_dim, cfg.feature_hidden_dim),
            nn.BatchNorm1d(cfg.feature_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.feature_hidden_dim, cfg.raw_feature_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
    
    def encode_trajectory(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode trajectory sequences [B, T_max, trajectory_input_dim] -> [B, trajectory_hidden_dim]"""
        packed = nn.utils.rnn.pack_padded_sequence(
            trajectories,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        _, (h_n, _) = self.trajectory_encoder_lstm(packed)
        last_hidden = h_n[-1]
        return self.trajectory_encoder_proj(last_hidden)
    
    def encode_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode raw features [B, raw_feature_dim] -> [B, feature_hidden_dim]"""
        return self.features_encoder(raw_features)
    
    def encode(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode to joint latent [B, joint_latent_dim]"""
        traj_encoded = self.encode_trajectory(trajectories, lengths)
        feat_encoded = self.encode_features(raw_features)
        fused = torch.cat([traj_encoded, feat_encoded], dim=1)
        return self.fusion(fused)
    
    def decode_trajectory(
        self,
        joint_latent: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Decode to trajectory [B, max_length, trajectory_input_dim]"""
        latent_seq = joint_latent.unsqueeze(1).repeat(1, max_length, 1)
        decoder_outputs, _ = self.trajectory_decoder_lstm(latent_seq)
        return self.trajectory_decoder_proj(decoder_outputs)
    
    def decode_features(self, joint_latent: torch.Tensor) -> torch.Tensor:
        """Decode to features [B, raw_feature_dim]"""
        return self.features_decoder(joint_latent)
    
    def decode(
        self,
        joint_latent: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode to both trajectory and features"""
        recon_trajectories = self.decode_trajectory(joint_latent, max_length)
        recon_features = self.decode_features(joint_latent)
        return recon_trajectories, recon_features
    
    def forward(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode"""
        max_length = trajectories.shape[1]
        joint_latent = self.encode(trajectories, lengths, raw_features)
        recon_trajectories, recon_features = self.decode(joint_latent, max_length)
        return joint_latent, recon_trajectories, recon_features

