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
            tokenizers={"observation": ObservationTokenizer},
            has_external_memory=False,
            device=args.device,
        )

        self.encoder = RawTextPolicyEncoder(encoder_config)
        self.decoder = EmmaPolicyDecoder(decoder_config)
        self.device = args.device

        self.apply(init_weights)

    def reset(self, n):
        self.encoded_manuals = None
        self.last_action = self.STAY_ACTION
        self.past_keys_values = self.decoder.reset(n)

    def tokenize_observations(self, observations: torch.tensor) -> torch.tensor:
        return self.decoder.tokenizer["observation"].encode(observations)

    def act(self, obs, temb, memory):
        if self.encoded_manuals is None:
            self.encoded_manuals = self.encoder(temb.unsqueeze(0))

        if (
            self.past_keys_values.size + self.decoder.tokens_per_block
            > self.decoder.config.max_tokens
        ):
            self.past_keys_values = self.decoder.reset(1)
            self.decoder(self.last_tokens, self.encoded_manuals, self.past_keys_values)

        act_tokens = (
            torch.tensor([self.last_action]).to(self.device).long().reshape(-1, 1, 1)
        )
        obs_tokens = self.tokenize_observations(obs.unsqueeze(1))
        tokens = rearrange(
            torch.cat((act_tokens, obs_tokens), dim=2), "b l k -> b (l k)"
        )
        self.last_tokens = tokens

        action_dists, _ = self.decoder(
            tokens, self.encoded_manuals, self.past_keys_values
        )
        action_dists = action_dists.squeeze()

        if self.training:
            dist = Categorical(action_dists)
            action = dist.sample()

            # store everything in memory
            memory.states.append(obs)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.texts.append(temb)
            action = action.item()
        else:
            action = torch.argmax(action_dists).item()
            """
            if random.random() < 0.05:  # random action with 0.05 prob
                action = random.randrange(0, self.action_dim)
            """

        self.last_action = action
        return action

    def evaluate(self, obs, action, ref_action, temb, mask):
        encoded_manuals = self.encoder(temb)

        obs_tokens = self.tokenize_observations(obs)
        act_tokens = rearrange(action, "b l -> b l 1")
        tokens = rearrange(
            torch.cat((act_tokens, obs_tokens), dim=2), "b l k -> b (l k)"
        )

        action_dists, value = self.decoder(tokens, encoded_manuals)
        """
        print(obs[0, 0].sum(-1))
        print(action_dists[0, 0])
        for i in range(action_dists.shape[1]):
            print("--->", action_dists[0, i].log()[ref_action[0, i]], mask[0, i])
        print(ref_action[0, 0])
        """

        action_dists = action_dists.reshape(-1, action_dists.shape[-1])
        value = value.flatten()

        dist = Categorical(action_dists)
        action_logprobs = dist.log_prob(ref_action.flatten())
        dist_entropy = dist.entropy()

        mask = mask.flatten()
        action_logprobs = torch.masked_select(action_logprobs, mask)
        value = torch.masked_select(value.squeeze(), mask)
        dist_entropy = torch.masked_select(dist_entropy, mask)

        return action_logprobs, value, dist_entropy


class RawTextPolicyEncoder(nn.Module):
   