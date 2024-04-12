from einops import rearrange
from typing import List

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from messenger.models.utils import BatchedEncoder

from .transformer import Transformer, TransformerConfig
from .slicer import Embedder


class BaseEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.tokenizer = {name: func() for name, func in config.tokenizers.items()}
        self.transformer = Transformer(config)


class StandardEncoder(BaseEncoder):
    """Transformer encoder for raw manuals
    Use BERT to embed the descriptions
    All descriptions are concatenated into a single sequence
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_encoder = BatchedEncoder(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device=config.device,
           