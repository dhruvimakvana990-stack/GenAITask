"""Named experiment presets for the bonus task (architecture comparison).

Each preset is a fully-formed :class:`ExperimentConfig`. They share the same
data source but vary depth, sequence length and capacity so their generated
text quality can be compared apples-to-apples.
"""

from __future__ import annotations

from .config import (
    DataConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    TrainingConfig,
)


def _experiments() -> dict[str, ExperimentConfig]:
    baseline = ExperimentConfig(
        name="baseline",
        data=DataConfig(sequence_length=80, step=4, tokenizer_mode="char",
                        max_chars=300_000),
        model=ModelConfig(embedding_dim=64, lstm_units=(256,), dropout=0.2),
        training=TrainingConfig(epochs=20, batch_size=128, model_name="baseline"),
        generation=GenerationConfig(num_tokens=400, temperature=0.8),
    )

    # Bonus 1: deeper, stacked LSTM -> more capacity to model long-range style.
    deep_lstm = ExperimentConfig(
        name="deep_lstm",
        data=DataConfig(sequence_length=100, step=3, tokenizer_mode="char"),
        model=ModelConfig(embedding_dim=96, lstm_units=(256, 256), dropout=0.3),
        training=TrainingConfig(epochs=25, batch_size=128, model_name="deep_lstm"),
        generation=GenerationConfig(num_tokens=400, temperature=0.7),
    )

    # Bonus 2: shorter context window -> faster but less coherent long-range.
    short_seq = ExperimentConfig(
        name="short_seq",
        data=DataConfig(sequence_length=40, step=3, tokenizer_mode="char"),
        model=ModelConfig(embedding_dim=64, lstm_units=(128,), dropout=0.2),
        training=TrainingConfig(epochs=20, batch_size=128, model_name="short_seq"),
        generation=GenerationConfig(num_tokens=400, temperature=0.8),
    )

    # Bonus 3: word-level tokenisation -> different unit of generation entirely.
    word_level = ExperimentConfig(
        name="word_level",
        data=DataConfig(sequence_length=20, step=1, tokenizer_mode="word",
                        max_chars=300_000),
        model=ModelConfig(embedding_dim=128, lstm_units=(256,), dropout=0.2),
        training=TrainingConfig(epochs=30, batch_size=128, model_name="word_level"),
        generation=GenerationConfig(num_tokens=80, temperature=0.8),
    )

    # "Proper" production-grade model: deeper + more data + longer context +
    # stronger regularisation. Designed to beat the baseline's val_loss and
    # generate noticeably more coherent text.
    proper = ExperimentConfig(
        name="proper",
        data=DataConfig(sequence_length=80, step=5, tokenizer_mode="char",
                        max_chars=450_000),
        model=ModelConfig(embedding_dim=96, lstm_units=(256, 256), dropout=0.3),
        training=TrainingConfig(epochs=30, batch_size=128,
                                early_stopping_patience=4, model_name="proper"),
        generation=GenerationConfig(num_tokens=500, temperature=0.7),
    )

    return {e.name: e for e in (baseline, deep_lstm, short_seq, word_level, proper)}


EXPERIMENTS = _experiments()


def get_experiment(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}")
    return EXPERIMENTS[name]
