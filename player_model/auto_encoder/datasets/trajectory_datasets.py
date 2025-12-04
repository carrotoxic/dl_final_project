import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from ..config import DataConfig


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: DataConfig, normalize: bool = True):
        self.cfg = cfg
        self.normalize = normalize
        self.files: List[Path] = sorted(cfg.data_root.glob("*.json"))
        if cfg.max_files is not None:
            self.files = self.files[:cfg.max_files]

        self.samples: List[Dict[str, Any]] = []
        for f in self.files:
            try:
                with f.open("r") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            trace = data.get(cfg.trace_key, None)
            if trace is None or len(trace) < cfg.min_length:
                continue

            stem = f.stem
            if "_" in stem:
                player_id = stem.split("_")[0]
            else:
                player_id = stem

            self.samples.append(
                {
                    "player_id": player_id,
                    "trace": trace,
                    "path": f,
                }
            )

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {cfg.data_root}")

        # Compute per-trajectory normalization statistics (min-max to [0, 1])
        # Each trajectory is normalized independently to [0, 1] per dimension using:
        #   normalized = (original - min) / (max - min)
        # This preserves the relative structure within each trajectory while making
        # the scale consistent across trajectories. The offset (min) and scale (max-min)
        # are saved per-trajectory to allow denormalization back to original coordinates.
        if self.normalize:
            for sample in self.samples:
                trace = np.array(sample["trace"], dtype=np.float32)
                # Compute min and max per dimension
                trace_min = trace.min(axis=0)  # [min_x, min_y]
                trace_max = trace.max(axis=0)  # [max_x, max_y]
                trace_range = trace_max - trace_min
                # Avoid division by zero - if range is 0, set scale to 1
                trace_range = np.where(trace_range < 1e-8, 1.0, trace_range)
                # Store offset (min) and scale (range) for denormalization
                sample["normalization_offset"] = torch.tensor(trace_min, dtype=torch.float32)
                sample["normalization_scale"] = torch.tensor(trace_range, dtype=torch.float32)
        else:
            # If not normalizing, set identity transforms
            for sample in self.samples:
                trace = np.array(sample["trace"], dtype=np.float32)
                sample["normalization_offset"] = torch.zeros(2, dtype=torch.float32)
                sample["normalization_scale"] = torch.ones(2, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        trace = sample["trace"]
        traj_tensor = torch.tensor(trace, dtype=torch.float32)
        
        # Get per-trajectory normalization parameters
        offset = sample["normalization_offset"]
        scale = sample["normalization_scale"]

        # Normalize the trajectory to [0, 1] per dimension
        if self.normalize:
            traj_tensor = (traj_tensor - offset) / scale

        return {
            "player_id": sample["player_id"],
            "trajectory": traj_tensor,
            "length": traj_tensor.shape[0],
            "path": sample["path"],
            "normalization_offset": offset,  # Store for denormalization
            "normalization_scale": scale,     # Store for denormalization
        }
    
    def denormalize(self, tensor: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to original scale using per-trajectory parameters.
        
        Args:
            tensor: Normalized tensor of shape [B, T, D] or [T, D]
            offset: Per-trajectory offset (min) of shape [B, D] or [D]
            scale: Per-trajectory scale (range) of shape [B, D] or [D]
        """
        return tensor * scale + offset


def trajectory_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    trajectories = [b["trajectory"] for b in batch]
    lengths = torch.tensor([t.shape[0] for t in trajectories], dtype=torch.long)
    player_ids = [b["player_id"] for b in batch]
    paths = [b["path"] for b in batch]
    
    # Collect normalization parameters for each trajectory
    offsets = torch.stack([b["normalization_offset"] for b in batch])  # [B, D]
    scales = torch.stack([b["normalization_scale"] for b in batch])    # [B, D]

    padded = pad_sequence(trajectories, batch_first=True)

    return {
        "trajectories": padded,
        "lengths": lengths,
        "player_ids": player_ids,
        "paths": paths,
        "normalization_offsets": offsets,  # [B, D]
        "normalization_scales": scales,     # [B, D]
    }