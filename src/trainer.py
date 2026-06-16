"""Training orchestration: callbacks, fit loop and history persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from .config import TrainingConfig
from .model import LSTMTextModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Drives ``model.fit`` with early stopping and best-checkpoint saving."""

    def __init__(self, model: LSTMTextModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        self.history: dict | None = None
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.config.checkpoint_dir) / f"{self.config.model_name}.keras"

    def _callbacks(self) -> list[tf.keras.callbacks.Callback]:
        return [
            # Restore the weights from the epoch with the lowest val_loss.
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            # Persist the best model so generation can run from disk later.
            callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            # Decay LR when progress plateaus -> better final convergence.
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ]

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        keras_model = self.model._require_built()
        logger.info("Training on %d samples, validating on %d", len(X_train), len(X_val))
        hist = keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=self._callbacks(),
            verbose=2,
        )
        self.history = hist.history
        self._save_history()
        return self.history

    def _save_history(self) -> None:
        out = Path(self.config.checkpoint_dir) / f"{self.config.model_name}_history.json"
        # Cast numpy floats so the JSON encoder is happy.
        serialisable = {k: [float(v) for v in vals] for k, vals in (self.history or {}).items()}
        out.write_text(json.dumps(serialisable, indent=2))
        logger.info("Saved training history to %s", out)
