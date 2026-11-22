"""Shared helpers for ds-sgen. Every other module imports from here."""

import json
import os
import random
import tempfile

import numpy as np
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(p: str) -> str:
    """Return p unchanged if absolute, else resolved against the repository root."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict.

    Relative entries of paths.{cache_dir, results_dir, hf_cache} are resolved
    against the repository root, so scripts work from any working directory
    on Windows and Linux (defaults: data/cache, data/results, data/model_cache).
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    for key in ("cache_dir", "results_dir", "hf_cache"):
        if paths.get(key):
            paths[key] = resolve_path(paths[key])
    return cfg


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is only needed for generation-side stages; re-plotting from cache works without it
        import torch
    except ImportError:
        return
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

    Prevents a corrupted cache if the job is interrupted mid-write.
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
