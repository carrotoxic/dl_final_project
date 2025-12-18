import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from ..config import DataConfig


def extract_timestep_features(timestep: Dict[str, Any]) -> np.ndarray:
    features = np.array([
        timestep.get("jump", 0),
        timestep.get("land", 0),
        timestep.get("bump", 0),
        timestep.get("kick", 0),
        timestep.get("kills_total", 0),
        timestep.get("kills_by_fire", 0),
        timestep.get("kills_by_stomp", 0),
        timestep.get("kills_by_shell", 0),
        timestep.get("kills_by_fall", 0),
        timestep.get("hurts", 0),
        timestep.get("collected", 0),
        timestep.get("death", 0),
    ], dtype=np.float32)
    return features


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.files: List[Path] = sorted(cfg.data_root.glob("*.json"))
        self.samples: List[Dict[str, Any]] = []
        
        for f in self.files:
            with f.open("r") as fp:
                data = json.load(fp)
            
            filename = f.stem
            player_id = filename.split("_")[0]
            
            trace = data.get("trace", [])
            if not trace:
                continue
            
            trace_features = [extract_timestep_features(ts) for ts in trace]
            last_timestep = trace[-1]
            
            completing_ratio = data.get("completing-ratio", 0.0)
            all_enemies = data.get("all_enemies", 1)
            all_coins = data.get("all_coins", 1)
            
            kills_ratio = data.get("#kills", 0) / all_enemies if all_enemies > 0 else 0.0
            kills_by_fire_ratio = data.get("#kills-by-fire", 0) / all_enemies if all_enemies > 0 else 0.0
            kills_by_stomp_ratio = data.get("#kills-by-stomp", 0) / all_enemies if all_enemies > 0 else 0.0
            kills_by_shell_ratio = data.get("#kills-by-shell", 0) / all_enemies if all_enemies > 0 else 0.0
            collected_coins_ratio = data.get("#coins", 0) / all_coins if all_coins > 0 else 0.0
            lives = data.get("lives", 0)
            
            total_jump = last_timestep.get("jump", 0)
            total_land = last_timestep.get("land", 0)
            total_kills = last_timestep.get("kills_total", 0)
            total_collected = last_timestep.get("collected", 0)
            total_hurts = last_timestep.get("hurts", 0)
            total_death = last_timestep.get("death", 0)
            
            overall_features = np.array([
                completing_ratio,
                kills_ratio,
                kills_by_fire_ratio,
                kills_by_stomp_ratio,
                kills_by_shell_ratio,
                collected_coins_ratio,
                lives,
                total_jump,
                total_land,
                total_kills,
                total_collected,
                total_hurts,
                total_death
            ], dtype=np.float32)

            self.samples.append(
                {
                    "player_id": player_id,
                    "path": f,
                    "trace_features": trace_features,
                    "overall_features": overall_features,
                }
            )

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {cfg.data_root}")
    
    @property
    def max_seq_len(self) -> int:
        """Compute the maximum sequence length in the dataset."""
        max_len = 0
        for sample in self.samples:
            trace_features = sample.get("trace_features")
            max_len = max(max_len, len(trace_features))
        return max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        trace_features = sample["trace_features"]
        trace_array = np.array(trace_features, dtype=np.float32)
        traj_tensor = torch.from_numpy(trace_array)
        
        mean = torch.zeros(traj_tensor.shape[1])
        std = torch.ones(traj_tensor.shape[1])
        
        if self.cfg.normalize:
            eps = 1e-8
            mean = traj_tensor.mean(axis=0)
            std = traj_tensor.std(axis=0)
            traj_tensor = (traj_tensor - mean) / (std + eps)

        return {
            "player_id": sample["player_id"],
            "path": sample["path"],
            "trajectory": traj_tensor,
            "length": traj_tensor.shape[0],
            "overall_features": sample["overall_features"],
            "normalization_mean": mean,
            "normalization_std": std,
        }
    
    def denormalize(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return tensor * (std + eps) + mean


def trajectory_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    player_ids = [b["player_id"] for b in batch]
    paths = [b["path"] for b in batch]

    trajectories = [b["trajectory"] for b in batch]
    padded = pad_sequence(trajectories, batch_first=True)
    lengths = torch.tensor([t.shape[0] for t in trajectories], dtype=torch.long)
    # Collect normalization parameters for each trajectory
    means = torch.stack([b["normalization_mean"] for b in batch])  # [B, D]
    stds = torch.stack([b["normalization_std"] for b in batch])    # [B, D]

    overall_features = torch.from_numpy(np.stack([b["overall_features"] for b in batch]).astype(np.float32))

    return {
        "player_ids": player_ids,
        "paths": paths,
        "trajectories": padded,
        "lengths": lengths,
        "normalization_means": means,  # [B, D]
        "normalization_stds": stds,     # [B, D]
        "overall_features": overall_features,
    }