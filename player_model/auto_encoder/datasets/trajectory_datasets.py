import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..config import DataConfig


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        trace = sample["trace"]
        traj_tensor = torch.tensor(trace, dtype=torch.float32)

        return {
            "player_id": sample["player_id"],
            "trajectory": traj_tensor,
            "length": traj_tensor.shape[0],
            "path": sample["path"],
        }


def trajectory_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    trajectories = [b["trajectory"] for b in batch]
    lengths = torch.tensor([t.shape[0] for t in trajectories], dtype=torch.long)
    player_ids = [b["player_id"] for b in batch]
    paths = [b["path"] for b in batch]

    padded = pad_sequence(trajectories, batch_first=True)

    return {
        "trajectories": padded,
        "lengths": lengths,
        "player_ids": player_ids,
        "paths": paths,
    }