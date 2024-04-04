import numpy as np
import torch

from messenger.models.utils import Encoder
from transformers import AutoModel, AutoTokenizer

from .oracle import TrainOracleWithMemory
from transformer_world_model.utils import ENTITY_IDS


class RolloutGenerator:
    def __init__(self, args):
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = Encoder(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device=args.device,
            max_length=36,
        )
        self.buffer = ObservationBuffer(
            device=args.device, buffer_size=args.emma_policy.hist_len
        )
        self.memory = Memory()

    @torch.no_grad()
    def generate(self, policy, env, split, games, num_episodes):
        self.memory.clear_memory()

        for game in games:
            for _ in range(num_episodes):
                # reset policy
                policy.reset(1)
                # reset environment
                obs, info = env.reset(split=split, entities=game)
                # reset obs