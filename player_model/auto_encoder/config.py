from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("data/player_trajectory_no_timeout_human")
    trace_key: str = "trace"
    min_length: int = 10
    max_files: int | None = None


@dataclass
class ModelConfig:
    model_name: str = "lstm"
    input_dim: int = 2
    hidden_dim: int =  32
    latent_dim: int = 8
    num_layers: int = 1


@dataclass
class FusionModelConfig:
    """Configuration for fusion autoencoder."""
    trajectory_latent_dim: int = 8  # From pre-trained LSTM autoencoder
    raw_feature_dim: int = 10  # status one-hot (3) + numeric features (7)
    joint_latent_dim: int = 16  # Joint latent space dimension
    hidden_dim: int = 32  # Hidden dimension for encoders/decoders
    fusion_hidden_dim: int = 32  # Hidden dimension for fusion layer
    dropout: float = 0.1
    use_attention: bool = False  # Use attention-based fusion or concatenation


@dataclass
class TrainConfig:
    batch_size: int = 1024
    num_epochs: int = 10000
    lr: float = 1e-3
    weight_decay: float = 0
    device: str = "cuda"
    log_interval: int = 50
    model_save_path: Path = Path("checkpoints/player_model/auto_encoder/lstm_ae_h32_z8_human.pt")
    gradient_clip: float = 1.0
    save_interval: int = 100


@dataclass
class FusionTrainConfig:
    """Training configuration for fusion autoencoder."""
    batch_size: int = 256
    num_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    log_interval: int = 50
    model_save_path: Path = Path("checkpoints/player_model/auto_encoder/fusion_ae_joint16.pt")
    gradient_clip: float = 1.0
    save_interval: int = 50
    # Loss weights
    trajectory_recon_weight: float = 1.0
    raw_recon_weight: float = 1.0


data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()
fusion_model_config = FusionModelConfig()
fusion_train_config = FusionTrainConfig()
