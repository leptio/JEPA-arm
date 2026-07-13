"""Transition dataset over collected .npz shards (directive §4.1 provenance preserved)."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths: Sequence[str], device: str = "cpu"):
        obs_t, act_t, obs_tp1, arm_id, src = [], [], [], [], []
        for i, p in enumerate(npz_paths):
            d = np.load(p)
            obs_t.append(d["obs_t"]); act_t.append(d["act_t"])
            obs_tp1.append(d["obs_tp1"]); arm_id.append(d["arm_id"])
            src.append(np.full((len(d["obs_t"]),), i, dtype=np.int64))
        self.obs_t = torch.from_numpy(np.concatenate(obs_t)).float()
        self.act_t = torch.from_numpy(np.concatenate(act_t)).float()
        self.obs_tp1 = torch.from_numpy(np.concatenate(obs_tp1)).float()
        self.arm_id = torch.from_numpy(np.concatenate(arm_id)).long()
        self.src = torch.from_numpy(np.concatenate(src)).long()
        self.paths = [str(p) for p in npz_paths]

    def __len__(self):
        return self.obs_t.shape[0]

    def __getitem__(self, i):
        return self.obs_t[i], self.act_t[i], self.obs_tp1[i], self.arm_id[i]


def find_shards(data_dir: str, arm: str | None = None,
                policy: str | None = None) -> list[str]:
    out = []
    for p in sorted(Path(data_dir).glob("*.npz")):
        name = p.name
        if arm is not None and not name.startswith(f"{arm}__"):
            continue
        if policy is not None and policy not in name:
            continue
        out.append(str(p))
    return out
