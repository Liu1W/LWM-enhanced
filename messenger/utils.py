import random
from typing import Dict
import json
import pickle
import logging

import torch
import numpy as np


class Policy:
    def __init__(self, policy):
        self.policy = policy

    def reset(self, obs, manual):
        pass

    def __call__(
        self, obs, buffer, manual, deterministic=False, temperature: float = False
    ):
        obs_hist = buffer.get_obs()
        return self.policy(
            obs_hist, manual, deterministic=deterministic, temperature=temperature
        )


def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility.

    Args:
        seed (int): seed to set
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_grid(obs: Dict[str, np.ndarray]) -> None:
    """Print the observation to terminal."""
    grid = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
    print(np.sum(grid, axis=-1).astype("uint8"))


def load_json(path: str) -> Dict:
    """Load json from path.

    Args:
        path (str): path to json file

    Returns