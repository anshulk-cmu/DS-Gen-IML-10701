"""Shared helpers for ds-sgen. Every other module imports from here."""

import json
import os
import random
import tempfile

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cache_path(cache_dir: str, name: str) -> str:
    """Return path for a named cache file: <cache_dir>/<name>.json."""
    return os.path.join(cache_dir, f"{name}.json")


def load_cache(path: str):
    """Load a JSON cache file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_cache(data, path: str):
    """Atomic JSON write: writes to a temp file then renames.

    Prevents corrupted cache if SLURM preempts mid-write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise
