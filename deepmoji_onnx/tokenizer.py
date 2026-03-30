"""Minimal DeepMoji tokenizer — pure Python + numpy, no PyTorch required.

Replicates the core logic of torchMoji's SentenceTokenizer / WordGenerator:
- lower-case
- URL → CUSTOM_URL
- @mention → CUSTOM_AT
- number token → CUSTOM_NUMBER
- unknown word → CUSTOM_UNKNOWN
- zero-pad / truncate to ``maxlen``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Regex patterns (simplified but consistent with torchMoji)
# ---------------------------------------------------------------------------

_RE_URL = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_RE_AT = re.compile(r"@\w+")
_RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_RE_SPLIT = re.compile(r"\s+")


def _preprocess(text: str) -> list[str]:
    """Tokenise a single sentence into a list of word-level tokens.

    Special tokens (CUSTOM_*) are injected before lowercasing so they survive
    the ``.lower()`` call intact.
    """
    # Inject special tokens first (regex are case-insensitive where needed)
    text = _RE_URL.sub(" CUSTOM_URL ", text)
    text = _RE_AT.sub(" CUSTOM_AT ", text)
    text = _RE_NUMBER.sub(" CUSTOM_NUMBER ", text)
    # Lowercase regular words but preserve CUSTOM_* tokens
    words = _RE_SPLIT.split(text.strip())
    return [w if w.startswith("CUSTOM_") else w.lower() for w in words if w]


class DeepMojiTokenizer:
    """Map text sentences to padded integer arrays.

    Args:
        vocab_path: Path to ``vocabulary.json`` (shipped alongside the model).
        maxlen:     Pad/truncate all sequences to this length (default 30).
        mask_value: Padding token id (default 0).
        unk_value:  Unknown word id (default 1).
    """

    # Special token ids (torchMoji convention)
    MASK = 0
    UNK = 1
    AT = 2
    URL = 3
    NUMBER = 4
    BREAK = 5

    _SPECIAL_MAP = {
        "CUSTOM_MASK":    MASK,
        "CUSTOM_UNKNOWN": UNK,
        "CUSTOM_AT":      AT,
        "CUSTOM_URL":     URL,
        "CUSTOM_NUMBER":  NUMBER,
        "CUSTOM_BREAK":   BREAK,
    }

    def __init__(
        self,
        vocab_path: Union[str, Path],
        maxlen: int = 30,
        mask_value: int = 0,
        unk_value: int = 1,
    ) -> None:
        self.maxlen = maxlen
        self.mask_value = mask_value
        self.unk_value = unk_value

        vocab_path = Path(vocab_path)
        with vocab_path.open(encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)

        # vocabulary.json maps word → index
        self.vocab: dict[str, int] = raw

    def tokenize(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of sentences to padded token arrays.

        Args:
            sentences: List of raw text strings.

        Returns:
            tokens:  int64 array (N, maxlen) — zero-padded.
            lengths: int64 array (N,) — number of non-padding tokens per row.
        """
        N = len(sentences)
        tokens = np.zeros((N, self.maxlen), dtype=np.int64)
        lengths = np.zeros(N, dtype=np.int64)

        for i, sent in enumerate(sentences):
            words = _preprocess(sent)
            ids = []
            for w in words:
                if w in self._SPECIAL_MAP:
                    ids.append(self._SPECIAL_MAP[w])
                elif w in self.vocab:
                    ids.append(self.vocab[w])
                else:
                    ids.append(self.unk_value)

            # truncate
            ids = ids[: self.maxlen]
            # at least length 1 to avoid empty sequence
            length = max(len(ids), 1)
            tokens[i, :len(ids)] = ids
            lengths[i] = length

        return tokens, lengths

    def __repr__(self) -> str:
        return (
            f"DeepMojiTokenizer(vocab_size={len(self.vocab)}, maxlen={self.maxlen})"
        )
