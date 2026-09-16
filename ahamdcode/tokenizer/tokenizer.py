from __future__ import annotations
from pathlib import Path
from .vocab import Vocabulary

class CodeTokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or Vocabulary()
        self.merges = merges or []
    @property
    def vocab_size(self):
        return self.vocab.size
    def encode(self, text, add_special=False):
        ids = []
        if add_special:
            ids.append(self.vocab.encode_token("<bos>"))
        for ch in text.split():
            ids.append(self.vocab.encode_token(ch))
        return ids or [self.vocab.encode_token("<unk>")]
    def decode(self, ids, skip_special=True):
        out = []
        special = {"<pad>", "<bos>", "<eos>", "<unk>"}
        for i in ids:
            if 0 <= i < len(self.vocab.tokens):
                tok = self.vocab.tokens[i]
                if skip_special and tok in special:
                    continue
                out.append(tok)
        return " ".join(out)
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        vocab = Vocabulary.load(directory / "vocab.json")
        return cls(vocab=vocab, merges=[])

def default_tokenizer_dir():
    from ahamdcode.utils.system import default_paths
    return default_paths()["tokenizer"]
