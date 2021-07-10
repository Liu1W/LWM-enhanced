from typing import List, Tuple
from pathlib import Path
import random

from vgdl.interfaces.gym import VGDLEnv
from torch.distributions import Categorical
import torch
import numpy as np

from messenger.envs.stage_two_custom import StageTwoCustom
from world_model.chatgpt_groundings.utils import parse_manuals
from world_model.transformer_world_model.utils import (
    process_parsed_manual_for_encoding,
    ENTITY_IDS,
)


class StageTwoTransformer(StageTwoCustom):
    """Stage Two Messenger environment which uses a world model
    to generate observations and rewards, instead of using the
    actual Messenger simulator.
    """

    def __init__(
        self,
        world_model,
        gpt_groundings=None,
        shuffle_obs: bool = False,
        fix_order: bool = False,
    ):
        super().__init__(shuffle_obs=shuffle_obs, fix_order=fix_order)

        self.world_model = world_model
        self.world_model.eval()

        assert self.world_model.manual_type in [
            "none",
            "emma",
            "standard",
            "standardv2",
            "direct",
            "oracle",
        ]

        self.gpt_groundings = gpt_groundings

    def _reset_world_model(self, action, obs, reward, done):
        past_keys_values = self.world_model.reset(action.shape