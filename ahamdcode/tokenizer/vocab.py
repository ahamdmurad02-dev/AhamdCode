from __future__ import annotations
import json
from pathlib import Path
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

class Vocabulary:
    def __init__(self, tokens=None):
        self.tokens = list(tokens or SPECIAL_TOKENS)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
    def encode_token(self, tok: str) -> int:
        return self.stoi.get(tok, self.stoi.get("<unk>", 3))
    @property
    def size(self):
        return len(self.tokens)
    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = data.get("tokens") or list(data.keys())
        return cls(tokens)
    def save(self, path):
        Path(path).write_text(json.dumps({"tokens": self.tokens}, indent=2), encoding="utf-8")
