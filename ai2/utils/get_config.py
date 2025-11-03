import json
from pathlib import Path

import flatdict

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
