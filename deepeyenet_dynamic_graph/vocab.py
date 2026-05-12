from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?|[.,;:!?()]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


@dataclass
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str, max_len: int) -> list[int]:
        if max_len <= 0:
            return []
        if max_len == 1:
            return [self.eos_id]
        ids = [self.bos_id]
        ids.extend(self.stoi.get(tok, self.unk_id) for tok in tokenize(text))
        ids.append(self.eos_id)
        ids = ids[:max_len]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        toks = []
        specials = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        for idx in ids:
            tok = self.itos[int(idx)]
            if skip_special and tok in specials:
                continue
            toks.append(tok)
        text = " ".join(toks)
        return text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        itos = list(data["itos"])
        return cls(stoi={tok: i for i, tok in enumerate(itos)}, itos=itos)


def build_vocab(texts: Iterable[str], min_freq: int = 1, max_size: int = 12000) -> Vocabulary:
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    words = [w for w, c in counter.most_common() if c >= min_freq and w not in specials]
    words = words[: max(0, max_size - len(specials))]
    itos = specials + words
    return Vocabulary(stoi={tok: i for i, tok in enumerate(itos)}, itos=itos)


def normalize_concept(concept: str) -> str:
    return " ".join(tokenize(concept))


def build_concepts(keyword_lists: Iterable[Iterable[str]], max_concepts: int = 128) -> list[str]:
    counter: Counter[str] = Counter()
    for kws in keyword_lists:
        counter.update(normalize_concept(k) for k in kws if normalize_concept(k))
    return [c for c, _ in counter.most_common(max_concepts)]
