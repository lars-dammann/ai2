import json
from pathlib import Path

import flatdict
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import v2

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs/configs.json", "r") as f:
        return json.load(f)


def get_config(new_config=None):
    config = load_config()

    if new_config is None:
        return config
    else:
        flat = flatdict.FlatDict(config)

        for new_key, new_value in new_config.items():
            update_keys = [key for key in flat.keys() if new_key in key.split(":")]
            if len(update_keys) > 0:
                for k in update_keys:
                    flat[k] = new_value
            else:
                flat[new_key] = new_value

        return flat.as_dict()

def get_mask(path, sample_id):
    mask = np.load(path / f"{sample_id}.npy").astype(bool)
    return torch.from_numpy(mask[np.newaxis, :, :])

def get_image(path, sample_id):
    scale = v2.ToDtype(torch.float32, scale=True)
    return scale(read_image(path / f"{sample_id}.png"))

def get_height(path, sample_id):
    return torch.from_numpy(
    (np.load(path / f"{sample_id}.npy")[np.newaxis, :, :]).astype(np.float32))
