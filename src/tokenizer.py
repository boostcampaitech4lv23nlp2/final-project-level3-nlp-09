from typing import List, Union

import html
import re

import ftfy
import torch
from transformers import AutoTokenizer


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


class HFTokenizer:
    "HuggingFace tokenizer wrapper"

    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts, return_tensors="pt", max_length=context_length, padding="max_length", truncation=True
        ).input_ids
        return input_ids


# class FoodTokenizer:
#     def __init__(self, tokenizer_name: str = "roberta-base"):
#         self.tokenizer_name = tokenizer_name

#     def get_tokenizer(self):
#         tokenizer = HFTokenizer(self.tokenizer_name)
#         return tokenizer


def get_tokenizer(tokenizer_name: str = "klue/roberta-base"):
    tokenizer = HFTokenizer(tokenizer_name)
    return tokenizer
