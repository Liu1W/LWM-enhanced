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
    # num tokens per entity
    tokens_per_entity = 3
    # number of observation tokens
    tokens_per_observation = 12

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.device = config.device
        self.tokenizer = {name: func() for name, func in config.tokenizers.items()}

        self._make_embedder()
        self._make_body()
        self._make_head()
        self._make_attention_masks()

    def _make_body(self):
        pass

    def _make_embedder(self):
        config = self.config

        n_o = self.tokens_per_observation
        n_r = 1

        token_types = ["action", "observation", "reward", "done"]
        vocab_size = {
            "action": 5,
            "observation": self.tokenizer["observation"].vocab_size,
            "reward": 10,
            "done": 2,
        }

        tokens_pattern = {}
        for t in token_types:
            tokens_pattern[t] = torch.zeros(config.tokens_per_block)
        # first token is action
        tokens_pattern["action"][0] = 1
        # observation tokens
        tokens_pattern["observation"][1 : (n_o + 1)] = 1
        # reward token
        tokens_pattern["reward"][(n_o + 1) : (n_o + n_r + 1)] = 1
        # done token
        tokens_pattern["done"][(n_o + n_r + 1) : (n_o + n_r + 2)] = 1

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks={t: tokens_pattern[t] for t in token_types},
            embedding_tables=nn.ModuleDict(
                {t: nn.Embedding(vocab_size[t], config.embed_dim) for t in token_types}
            ),
        )
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

    def _make_head(self):
        config = self.config

        n_o = self.tokens_per_observation
        n_r = 1

        head_token_types = ["observation", "reward", "done"]
        vocab_size = {
            "observation": self.tokenizer["observation"].vocab_size,
            "reward": 10,
            "done": 2,
        }

        head_tokens_pattern = {}
        for t in head_token_types:
            head_tokens_pattern[t] = torch.zeros(config.tokens_per_block)
        head_tokens_pattern["observation"][:n_o] = 1
        head_tokens_pattern["reward"][n_o : (n_o + n_r)] = 1
        head_tokens_pattern["done"][(n_o + n_r) : (n_o + n_r + 1)] = 1

        self.head = nn.ModuleDict(
            {
                t: Head(
                    max_blocks=config.max_blocks,
                    block_mask=head_tokens_pattern[t],
                    head_module=nn.Sequential(
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, vocab_size[t]),
                    ),
                )
                for t in head_token_types
            }
        )

    def _make_attention_masks(self):
        history_masks = torch.tril(
            torch.ones(self.config.max_tokens, self.config.max_tokens)
        )
        self.register_buffer("history_masks", history_masks)

    def reset(self, n: int) -> KeysValues:
        return self.decoder.generate_empty_keys_values(n, self.config.max_tokens)


class StandardDecoder(BaseDecoder):
    """Standard Transformer decoder with interleaved multi-headed self- and cross-attentions"""

    def __init__(self, config: TransformerConfig):
        super().__i