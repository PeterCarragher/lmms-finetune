from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

import torch
from transformers import PreTrainedTokenizer, AutoProcessor


class BaseDataCollator(ABC, object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self, 
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[AutoProcessor] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
    
    @property
    def IGNORE_TOKEN_ID(self) -> int:
        return -100

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    @abstractmethod
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]: ...