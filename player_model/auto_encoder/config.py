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
    frame_hidden_dim: int = 32
    hidden_dim: int =  32
    latent_dim: int = 16
    num_layers: int = 1
    use_batch_norm: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 0
    device: str = "cuda"
    log_interval: int = 50
    model_save_path: Path = Path("checkpoints/player_model/auto_encoder/lstm_ae.pt")
    gradient_clip: float = 1.0
    save_interval: int = 100


data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()