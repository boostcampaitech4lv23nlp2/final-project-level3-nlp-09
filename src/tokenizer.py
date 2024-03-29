from typing import List, Union

import html
import itertools
import json
import re
from itertools import product

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


def get_tokenizer(tokenizer_name: str = "klue/roberta-base"):
    tokenizer = HFTokenizer(tokenizer_name)
    return tokenizer


class FoodTokenizer:
    def __init__(
        self,
        path,
        remove_string=["_", "(", ")", "*", " ", "!", "?", "-"],
        replace=None,
        merge_strings=None,
        configs=None,
    ):
        with open(path, "r") as f:
            self.tokens_by_length = json.load(f)
        self.tokens_by_length = {int(k): v for k, v in self.tokens_by_length.items()}
        self.total = 0
        self.tokens = []
        self.lengths = sorted(self.tokens_by_length.keys(), reverse=True)
        self.dict = {"unk": 0, "sos": 1, "pad": 2}
        self.replace = {
            "고춧": "고추",
            "머릿": "머리",
            "배춧": "배추",
            "조갯": "조개",
            "김칫": "김치",
            "양팟": "양파",
            "북엇": "북어",
            "챗": "채",
            "만둣": "만두",
            "고깃": "고기",
        }
        self.configs = configs

        if isinstance(replace, dict):
            self.replace.update(replace)

        if isinstance(remove_string, (list, tuple)):
            self.patterns = "[" + "".join(remove_string) + "]"

        if self.lengths[-1] == 0:
            self.lengths = self.lengths[:-1]
        for l in self.lengths:
            self.tokens.append(self.tokens_by_length[l])
            self.total += len(self.tokens_by_length[l])
            for t in sorted(self.tokens_by_length[l]):
                self.dict[t] = len(self.dict)

        if isinstance(merge_strings, (list, tuple)):
            for s in merge_strings:
                self.dict[s] = len(self.dict)
                self.total += 1
                self.tokens[len(s) - 1].append(s)
        self.dict["eos"] = len(self.dict)
        self.total += 4
        self.cache = {}

    def __call__(self, names):
        assert isinstance(names, (list, tuple))
        ids = []
        max_length = self.configs["text_cfg"]["context_length"] if self.configs else 0
        for name in names:
            if hasattr(self, "patterns"):
                name = re.sub(self.patterns, "", name)
            if name not in self.cache:
                tokens = self.priority(self.filter(self.v2(name)))[0]
                tokens = [self.dict["sos"]] + [t[1] for t in tokens] + [self.dict["eos"]]
                self.cache[name] = tokens
            else:
                tokens = self.cache[name]
            if len(tokens) > max_length:
                max_length = len(tokens)
            ids.append(tokens)

        for i in range(len(ids)):
            if max_length - len(ids[i]) <= 0:
                continue
            ids[i] = ids[i] + [self.dict["pad"] for _ in range(max_length - len(ids[i]))]

        ids = self.get_list_to_tensor(ids)
        return ids

    def v2(self, name):
        results = []
        if name == "":
            return [[]]
        for tokens_by_length, length in zip(self.tokens, self.lengths):
            if length > len(name):
                continue
            for token in tokens_by_length:
                for i in range(len(name) - length + 1):
                    if name[i : i + length] == token:
                        for l, r in product(self.v2(name[:i]), self.v2(name[i + length :])):
                            results.append(l + [token] + r)
        if len(results) == 0:
            return [[name]]
        else:
            return results

    def filter(self, results):
        results = list(k for k, _ in itertools.groupby(sorted(results)))
        new_results = []
        for r in results:
            tmp = []
            for word in r:
                tmp.append((word, self.dict.get(word, 0)))
            new_results.append(tmp)
        return new_results

    def priority(self, results):
        return sorted(
            results,
            key=lambda x: (len(x), sum([1 for r in x if r[1] == 0])),
        )

    def get_list_to_tensor(self, list):
        tensor = torch.tensor(list)
        return tensor
