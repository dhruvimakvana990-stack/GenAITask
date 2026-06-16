"""Configuration objects for the LSTM text-generation pipeline.

Centralising every tunable knob in immutable dataclasses keeps the rest of the
code free of "magic numbers" and makes experiments (the bonus task) a matter of
swapping a config object rather than editing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    """Where the raw text comes from and how it is sliced/encoded."""

    source_path: str = "data/shakespeare.txt"
    # Tokenisation granularity: character-level is fast and robust on CPU,
    # word-level produces more "word-like" output but needs a larger model.
    tokenizer_mode: Literal["char", "word"] = "char"
    # Cap the corpus so training stays feasible on a CPU. None = use everything.
    max_chars: int | None = 350_000
    # Length of the input window the model sees before predicting the next token.
    sequence_length: int = 100
    # Slide the window by this many tokens when building (input, target) pairs.
    step: int = 3
    # Fraction of generated sequences held out for validation.
    validation_split: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """LSTM architecture hyper-parameters."""

    embedding_dim: int = 64
    # One entry per stacked LSTM layer; the list length controls model depth.
    lstm_units: tuple[int, ...] = (256,)
    dropout: float = 0.2
    recurrent_dropout: float = 0.0  # 0.0 keeps the fast cuDNN/CPU kernel.


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, batching and regularisation settings."""

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    # Stop early when validation loss stops improving for this many epochs.
    early_stopping_patience: int = 3
    checkpoint_dir: str = "models"
    model_name: str = "lstm_textgen"


@dataclass(frozen=True)
class GenerationConfig:
    """Controls the sampling behaviour at generation time."""

    num_tokens: int = 400
    # Softmax temperature: <1.0 = conservative, >1.0 = more adventurous.
    temperature: float = 0.8


@dataclass(frozen=True)
class ExperimentConfig:
    """Aggregates the four sub-configs into a single named experiment."""

    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict:
        """Flatten the config tree for logging / serialisation."""
        return asdict(self)
