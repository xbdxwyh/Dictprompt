from dataclasses import dataclass

from transformers.modeling_outputs import (
    ModelOutput
)
import torch
from typing import Optional, Tuple


@dataclass
class WordEmbdPairOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
    """
    embd_word1: Optional[Tuple[torch.FloatTensor]] = None
    embd_word2: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
