import os
import yaml
import numpy as np


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)


def load_yaml_config(file_path: str) -> Config:
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
