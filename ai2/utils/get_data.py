"""Data loading and configuration helpers for the ai2 package."""

import json
from pathlib import Path

import flatdict
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import v2


def load_config(file):
    """Load a JSON config file from the repository configs directory.

    Args:
        file: Optional config filename in the configs directory.

    Returns:
        Parsed configuration dictionary.
    """
    with open(file, "r") as f:
        return json.load(f)


def get_config(file, new_config=None):
    """Load config data and optionally override nested values.

    Args:
        new_config: Optional key-value overrides applied to nested config keys.
        file: Optional config filename in the configs directory.

    Returns:
        Configuration dictionary with optional overrides applied.
    """
    config = load_config(file)

    if new_config is None:
        return config

    return update_config(config, new_config)

def update_config(config, new_config):
    """Update a configuration dictionary with new values.

    Args:
        config: Original configuration dictionary.
        new_config: Dictionary of new values to update the original config.

    Returns:
        Updated configuration dictionary.
    """
    flat = flatdict.FlatDict(config)

    for new_key, new_value in new_config.items():
        update_keys = [key for key in flat.keys() if new_key in key.split(":")]
        if len(update_keys) > 0:
            for key in update_keys:
                flat[key] = new_value
        else:
            flat[new_key] = new_value

    return flat.as_dict()


def get_mask(path, sample_id):
    """Load a mask as a boolean tensor with a channel dimension.

    Args:
        path: Directory containing mask files.
        sample_id: Sample identifier without file extension.

    Returns:
        Boolean tensor with shape (1, H, W).
    """
    mask = np.load(path / f"{sample_id}.npy").astype(bool)
    return torch.from_numpy(mask[np.newaxis, :, :])


def get_image(path, sample_id):
    """Load an RGB image as a float32 tensor scaled to [0, 1].

    Args:
        path: Directory containing image files.
        sample_id: Sample identifier without file extension.

    Returns:
        Float tensor with shape (3, H, W).
    """
    scale = v2.ToDtype(torch.float32, scale=True)
    return scale(read_image(path / f"{sample_id}.png"))


def get_height(path, sample_id):
    """Load a height map as a float32 tensor with a channel dimension.

    Args:
        path: Directory containing height-map files.
        sample_id: Sample identifier without file extension.

    Returns:
        Float tensor with shape (1, H, W).
    """
    return torch.from_numpy(
        np.load(path / f"{sample_id}.npy")[np.newaxis, :, :].astype(np.float32)
    )
