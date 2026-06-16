"""End-to-end pipeline entrypoint: preprocess -> build -> train -> generate.

Usage
-----
    python main.py train                 # train the baseline experiment
    python main.py generate --seed "to be or not to be"
    python main.py train --experiment deep_lstm

The ``Pipeline`` class wires the components together so that ``main`` stays a
thin CLI layer over a reusable object.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import ExperimentConfig
from src.experiments import EXPERIMENTS, get_experiment
from src.generator import TextGenerator
from src.model import LSTMTextModel
from src.preprocessing import Corpus
from src.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


class Pipeline:
    """Owns one experiment config and the artefacts produced from it."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.corpus = Corpus(config.data)

    # -- training ---------------------------------------------------------
    def train(self) -> None:
        self.corpus.load()
        X_train, y_train, X_val, y_val = self.corpus.make_datasets()

        model = LSTMTextModel(
            vocab_size=self.corpus.tokenizer.vocab_size,
            sequence_length=self.config.data.sequence_length,
            model_cfg=self.config.model,
            train_cfg=self.config.training,
        ).build()
        logger.info("Model architecture:\n%s", model.summary())

        trainer = ModelTrainer(model, self.config.training)
        trainer.train(X_train, y_train, X_val, y_val)
        self._save_tokenizer()
        logger.info("Training complete. Best model at %s", trainer.checkpoint_path)

    # -- generation -------------------------------------------------------
    def generate(self, seeds: list[str]) -> dict[str, str]:
        self.corpus.load()  # rebuilds the identical vocabulary deterministically
        ckpt = Path(self.config.training.checkpoint_dir) / f"{self.config.training.model_name}.keras"
        keras_model = LSTMTextModel.load(ckpt)

        generator = TextGenerator(
            keras_model, self.corpus.tokenizer, self.config.data.sequence_length
        )
        results: dict[str, str] = {}
        for seed in seeds:
            text = generator.generate(seed, self.config.generation)
            results[seed] = text
            logger.info("\n--- Seed: %r ---\n%s\n", seed, text)
        self._save_samples(results)
        return results

    # -- helpers ----------------------------------------------------------
    def _save_tokenizer(self) -> None:
        out = Path(self.config.training.checkpoint_dir) / f"{self.config.training.model_name}_vocab.json"
        out.write_text(json.dumps(self.corpus.tokenizer.token_to_id, ensure_ascii=False))
        logger.info("Saved vocabulary to %s", out)

    def _save_samples(self, results: dict[str, str]) -> None:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"samples_{self.config.name}.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("Saved generated samples to %s", out)


DEFAULT_SEEDS = [
    "to be or not to be",
    "all the world is a stage",
    "shall i compare thee",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM Shakespeare text generator")
    parser.add_argument("command", choices=["train", "generate"], help="action to run")
    parser.add_argument("--experiment", default="baseline",
                        choices=list(EXPERIMENTS), help="named experiment config")
    parser.add_argument("--seed", action="append", help="seed text (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_experiment(args.experiment)
    pipeline = Pipeline(config)

    if args.command == "train":
        pipeline.train()
    else:
        pipeline.generate(args.seed or DEFAULT_SEEDS)


if __name__ == "__main__":
    main()
