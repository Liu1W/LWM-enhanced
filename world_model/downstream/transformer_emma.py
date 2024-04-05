""" EMMA policy with Transformer base architecture"""

from typing import Any, Optional, Tuple, List
from copy import deepcopy as dc
from einops import rearrange

from torch.distributions import Categorical
import torch
import torch.nn as nn

from transformer_world_model.tokenizer import ObservationTokenizer
from transformer_world_model.transformer import (
    TransformerConfig,
    Transformer,
    ExternalAttention,
)
from transformer_world_model.slicer import Embedder, Head
from transformer_world_model.kv_caching import KeysValues
from transformer_world_model.world_model import init_weights


class TransformerEMMA(nn.Module):
    STAY_ACTION = 4

    def __init__(self, args):
        super().__init__()

        encoder_config = TransformerConfig(
            tokens_per_block=args.encoder_tokens_per_block,
            max_blocks=args.encoder_max_blocks,
            num_layers=args.encoder_layers,
            num_heads=args.encoder_num_heads,
            embed_dim=args.hidden_size,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            tokenizers={},
            has_external_memory=False,
            device=args.device,
        )

        decoder_config = TransformerConfig(
            tokens_per_block=args.decoder_tokens_per_block,
            max_blocks=args.decoder_max_blocks,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_num_heads,
            embed_dim=args.hidden_size,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            tokenizers={"observation": ObservationTo