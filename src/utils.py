from typing import Dict

import yaml


def read_yaml(path: str) -> Dict:
    with open(path, 'rb') as f:
        return yaml.safe_load(f)
