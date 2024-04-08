from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from copy import deepcopy as dc
from einops import rearrange

import torch
import torch.nn as nn

from .transformer import Transformer, TransformerConfig, ExternalAttention
from .kv_caching import KeysValues
from .slicer import Embedder, Head


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_dones: torch.FloatTensor


class BaseDecoder(nn.Module):
    # num toke