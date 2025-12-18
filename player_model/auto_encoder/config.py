from dataclasses import dataclass
from pathlib import Path
from abc import ABC
from typing import Literal, Union


@dataclass
class DataConfig:
    data_root: Path = Path("data/player_trajectory_cumulative_no_timeout_human")
    normalize: bool = True


@dataclass
class BaseModelConfig(ABC):
    """Abstract base class for model configurations."""
    model_name: str = "lstm"


@dataclass
class LSTMModelConfig(BaseModelConfig):
    """Configuration for LSTM autoencoder."""
    model_name: str = "lstm"
    input_dim: int = 12
    hidden_dim: int = 32
    latent_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.1


@dataclass
class TransformerModelConfig(BaseModelConfig):
    """Configuration for transformer autoencoder."""
    model_name: str = "transformer"
    input_dim: int = 12
    hidden_dim: int = 32
    latent_dim: int = 8
    num_heads: int = 4
    ff_dim: int = 32
    dropout: float = 0.1


ModelConfig = Union[LSTMModelConfig, TransformerModelConfig]


def create_model_config(
    model_name: Literal["lstm", "transformer"] = "lstm",
    **kwargs
) -> ModelConfig:
    """Factory function to create model config based on model_name."""
    if model_name == "lstm":
        return LSTMModelConfig(**kwargs)
    elif model_name == "transformer":
        return TransformerModelConfig(**kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Must be one of: lstm, transformer")


@dataclass
class TrainConfig:
    model_name: str = "transformer"
    batch_size: int = 128
    num_epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0
    device: str = "cuda"
    log_interval: int = 50
    model_save_path: Path = Path(f"checkpoints/{model_name}/{model_name}.pt")
    gradient_clip: float = 1.0
    trajectory_recon_weight: float = 1.0
    latent_recon_weight: float = 1.0
