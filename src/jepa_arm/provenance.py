"""Provenance & reproducibility bookkeeping (directive §4.1, §7.1, §7.2).

Every run writes a provenance.json capturing: seeds, config hashes (safety + prereg),
behavior-policy identity, dependency versions, git-less content hashes, and any
scale_notes documenting reductions from the mandated scale (HONESTY.md §3).

A run whose safety config is not recorded is INVALID and must be discarded (§3.5); this
module makes that record mandatory by construction (RunContext requires the hash).
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def dep_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "mujoco", "torch", "gymnasium"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "absent"
    try:
        import torch
        out["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        pass
    return out


@dataclass
class RunContext:
    """One logged, reproducible run (directive §7.2: every claim traces to one of these)."""
    run_id: str
    seed: int
    arm: str
    regime: str                    # e.g. "per_arm", "cross_embodiment"
    method: str                    # e.g. "bridge", "cem_mpc", "value_fn", "rrt", ...
    behavior_policy: str           # data-collection policy identity (§2.3.1, §4.1)
    safety_config_path: str
    prereg_config_path: str
    out_dir: str
    seed_record: dict = field(default_factory=dict)
    scale_notes: dict = field(default_factory=dict)   # HONESTY.md §3 reductions
    extra: dict = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        d["safety_config_sha256_16"] = file_hash(self.safety_config_path)
        d["prereg_config_sha256_16"] = file_hash(self.prereg_config_path)
        d["deps"] = dep_versions()
        d["argv"] = sys.argv
        d["wall_clock_s"] = round(time.time() - self._t0, 3)
        d["sim_only"] = True        # see HONESTY.md §1
        return d

    def write(self) -> Path:
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "provenance.json"
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p
