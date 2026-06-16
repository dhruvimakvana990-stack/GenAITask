"""Iterative text generation with temperature-controlled sampling."""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf

from .config import GenerationConfig
from .preprocessing import TextCleaner, Tokenizer

logger = logging.getLogger(__name__)


class TextGenerator:
    """Generates text autoregressively from a trained model + tokenizer."""

    def __init__(self, keras_model: tf.keras.Model, tokenizer: Tokenizer,
                 sequence_length: int) -> None:
        self.model = keras_model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    @staticmethod
    def _sample(probs: np.ndarray, temperature: float) -> int:
        """Temperature sampling: reshape the distribution then draw from it.

        temperature -> 0 approaches greedy/argmax (repetitive but safe);
        higher temperature flattens the distribution (more diverse, riskier).
        """
        if temperature <= 0:
            return int(np.argmax(probs))
        logits = np.log(np.clip(probs, 1e-10, 1.0)) / temperature
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, seed: str, config: GenerationConfig) -> str:
        """Produce ``config.num_tokens`` new tokens following ``seed``."""
        # Clean the seed the same way the training corpus was cleaned.
        cleaned = TextCleaner.clean(seed)
        seed_ids = list(self.tokenizer.encode(cleaned))
        if not seed_ids:
            raise ValueError("Seed produced no known tokens after cleaning.")

        ids = list(seed_ids)
        generated_ids: list[int] = []
        for _ in range(config.num_tokens):
            # Left-pad / truncate the context window to the fixed input length.
            window = ids[-self.sequence_length:]
            padded = [0] * (self.sequence_length - len(window)) + window
            x = np.array(padded, dtype=np.int32)[None, :]

            probs = self.model.predict(x, verbose=0)[0]
            next_id = self._sample(probs, config.temperature)

            ids.append(next_id)
            generated_ids.append(next_id)

        # Return seed + continuation as a single decoded string.
        return self.tokenizer.decode(seed_ids + generated_ids)
