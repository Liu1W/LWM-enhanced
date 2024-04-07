import sys

sys.path.append("..")
import random
import json
from collections import defaultdict
from pprint import pprint

import torch
import numpy as np


from messenger.envs import make_env
from chatgpt_groundings.utils import load_gpt_groundings
import flags
import train_wm as utils

args = flags.make()
world_model = utils.make_model(args)

gpt_groundings = load_gpt_groundings(args)

env = make_env(
    args, world_mod