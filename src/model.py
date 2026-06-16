"""LSTM model definition, wrapped in a class that owns its Keras graph.

``LSTMTextModel`` encapsulates construction, compilation, save and load so the
trainer and generator only ever talk to a clean object interface rather than
raw Keras calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class LSTMTextModel:
    """An embedding -> stacked-LSTM -> softmax next-token classifier."""

    def __init__(self, vocab_size: int, sequence_length: int,
                 model_cfg: ModelConfig, train_cfg: TrainingConfig) -> None:
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.keras_model: tf.keras.Model | None = None

    def build(self) -> "LSTMTextModel":
        """Assemble and compile the network described by ``ModelConfig``."""
        cfg = self.model_cfg
        net = models.Sequential(name="lstm_textgen")
        # Embedding turns integer token ids into dense trainable vectors.
        net.add(layers.Input(shape=(self.sequence_length,)))
        net.add(layers.Embedding(self.vocab_size, cfg.embedding_dim))

        # Stack LSTM layers; every layer but the last must return sequences so
        # the next LSTM receives a time series rather than a single vector.
        for i, units in enumerate(cfg.lstm_units):
            is_last = i == len(cfg.lstm_units) - 1
            net.add(layers.LSTM(
                units,
                return_sequences=not is_last,
                dropout=cfg.dropout,
                recurrent_dropout=cfg.recurrent_dropout,
            ))

        net.add(layers.Dropout(cfg.dropout))
        # Dense + softmax over the vocabulary = probability of each next token.
        net.add(layers.Dense(self.vocab_size, activation="softmax"))

        net.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(self.train_cfg.learning_rate),
            metrics=["accuracy"],
        )
        self.keras_model = net
        logger.info("Built model with %d parameters", net.count_params())
        return self

    def summary(self) -> str:
        lines: list[str] = []
        self._require_built().summary(print_fn=lines.append)
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        self._require_built().save(path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> tf.keras.Model:
        logger.info("Loading model from %s", path)
        return models.load_model(path)

    def _require_built(self) -> tf.keras.Model:
        if self.keras_model is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        return self.keras_model
