"""Text loading, cleaning, tokenisation and (input, target) pair construction.

Design notes
------------
* ``TextCleaner``  -> single responsibility: normalise raw text.
* ``Tokenizer``    -> abstract base defining the encode/decode contract;
  ``CharTokenizer`` and ``WordTokenizer`` are interchangeable implementations
  (polymorphism), so the rest of the pipeline never branches on the mode.
* ``SequenceBuilder`` -> turns an encoded stream into sliding-window training
  pairs and a train/validation split.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import DataConfig

logger = logging.getLogger(__name__)


class TextCleaner:
    """Normalises raw text: lowercase + punctuation removal (per the task)."""

    # Keep word characters and whitespace; drop everything else.
    _PUNCT_RE = re.compile(r"[^a-z0-9\s]")
    _WHITESPACE_RE = re.compile(r"\s+")

    @classmethod
    def clean(cls, text: str) -> str:
        text = text.lower()
        text = cls._PUNCT_RE.sub(" ", text)
        text = cls._WHITESPACE_RE.sub(" ", text)
        return text.strip()


class Tokenizer(ABC):
    """Abstract token<->id codec. Subclasses define the unit of tokenisation."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Break raw text into the atomic tokens for this strategy."""

    @abstractmethod
    def join(self, tokens: list[str]) -> str:
        """Inverse of :meth:`split` for human-readable output."""

    def fit(self, text: str) -> "Tokenizer":
        """Build the vocabulary from a corpus. Returns self for chaining."""
        tokens = sorted(set(self.split(text)))
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        logger.info("Built %s vocabulary of %d tokens", type(self).__name__, self.vocab_size)
        return self

    def encode(self, text: str) -> np.ndarray:
        return np.array(
            [self.token_to_id[t] for t in self.split(text) if t in self.token_to_id],
            dtype=np.int32,
        )

    def decode(self, ids: list[int]) -> str:
        return self.join([self.id_to_token[i] for i in ids])


class CharTokenizer(Tokenizer):
    """Character-level tokeniser (small vocab, fast to train)."""

    def split(self, text: str) -> list[str]:
        return list(text)

    def join(self, tokens: list[str]) -> str:
        return "".join(tokens)


class WordTokenizer(Tokenizer):
    """Word-level tokeniser (larger vocab, more word-like generations)."""

    def split(self, text: str) -> list[str]:
        return text.split()

    def join(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def build_tokenizer(mode: str) -> Tokenizer:
    """Factory: map a config string to the matching tokeniser implementation."""
    tokenizers = {"char": CharTokenizer, "word": WordTokenizer}
    if mode not in tokenizers:
        raise ValueError(f"Unknown tokenizer_mode '{mode}'. Choose from {list(tokenizers)}.")
    return tokenizers[mode]()


class SequenceBuilder:
    """Creates sliding-window (input, next-token) pairs and a train/val split."""

    def __init__(self, sequence_length: int, step: int, validation_split: float) -> None:
        self.sequence_length = sequence_length
        self.step = step
        self.validation_split = validation_split

    def build(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised window extraction -> X: (n, seq_len), y: (n,)."""
        seq_len, step = self.sequence_length, self.step
        n_windows = (len(encoded) - seq_len) // step
        if n_windows <= 0:
            raise ValueError(
                f"Corpus too short ({len(encoded)} tokens) for sequence_length={seq_len}."
            )
        # Index gymnastics build every window without a Python loop.
        starts = np.arange(n_windows) * step
        offsets = np.arange(seq_len)
        X = encoded[starts[:, None] + offsets]      # (n_windows, seq_len)
        y = encoded[starts + seq_len]               # (n_windows,)
        logger.info("Built %d sequences of length %d", len(X), seq_len)
        return X, y

    def train_val_split(
        self, X: np.ndarray, y: np.ndarray, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        n_val = int(len(X) * self.validation_split)
        return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


class Corpus:
    """High-level facade tying cleaning, tokenising and windowing together."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.tokenizer: Tokenizer = build_tokenizer(config.tokenizer_mode)
        self.raw_text: str = ""
        self.clean_text: str = ""

    def load(self) -> "Corpus":
        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. See README for download instructions."
            )
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self.config.max_chars is not None:
            text = text[: self.config.max_chars]
        self.raw_text = text
        self.clean_text = TextCleaner.clean(text)
        self.tokenizer.fit(self.clean_text)
        logger.info("Loaded corpus: %d raw chars -> %d clean chars",
                    len(self.raw_text), len(self.clean_text))
        return self

    def make_datasets(self):
        """Return (X_train, y_train, X_val, y_val) ready for the model."""
        encoded = self.tokenizer.encode(self.clean_text)
        builder = SequenceBuilder(
            self.config.sequence_length, self.config.step, self.config.validation_split
        )
        X, y = builder.build(encoded)
        return builder.train_val_split(X, y)
