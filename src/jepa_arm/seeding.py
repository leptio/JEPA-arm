"""Determinism utilities (directive §4.4). Every run fixes and records all seeds."""
from __future__ import annotations
import os
import random
import numpy as np

try:
    import torch
except Exception:  # torch optional for pure-classical paths
    torch = None


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict:
    """Seed Python, NumPy, and Torch (+CUDA). Returns a record for provenance."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    record = {"seed": int(seed), "python_hashseed": str(seed)}
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuDNN determinism knobs; recorded so a reader knows the exact setting.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        record.update(
            torch_version=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cudnn_deterministic=bool(deterministic_torch),
        )
    return record


def rng(seed: int) -> np.random.Generator:
    """A local NumPy Generator for reproducible, side-effect-free sampling."""
    return np.random.default_rng(seed)
