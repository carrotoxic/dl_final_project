from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("data/player_trajectory")
    trace_key: str = "trace"
    min_length: int = 10
    max_files: int | None = None


@dataclass
class ModelConfig:
    input_dim: int = 2
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    log_interval: int = 50
    model_save_path: Path = Path("checkpoints/player_model/auto_encoder/lstm_ae.pt")


data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()