"""Data loading utilities for ds-sgen experiments."""

import json
import os
from pathlib import Path
from datasets import load_dataset


def load_questions(dataset_name: str, split: str = "test", cache_dir: str = None):
    """Load a QA dataset and return list of question dicts."""
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    return ds


def load_cached(path: str):
    """Load a JSON cache file if it exists."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_cached(data, path: str):
    """Save data to a JSON cache file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
