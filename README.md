# LSTM Text Generation (Generative AI Task)

An LSTM-based text generator trained on **Shakespeare's Complete Works**. The
project covers the full pipeline required by the task: data preprocessing,
model design, training with checkpoints/early-stopping, and iterative text
generation from a seed — built with a clean, object-oriented architecture.

## Architecture

The code follows separation-of-concerns with one responsibility per module:

```
GenAITask/
├── main.py                 # CLI entrypoint + Pipeline orchestrator
├── src/
│   ├── config.py           # Immutable dataclass configs (no magic numbers)
│   ├── preprocessing.py    # TextCleaner, Tokenizer (Char/Word), Corpus, SequenceBuilder
│   ├── model.py            # LSTMTextModel — owns the Keras graph
│   ├── trainer.py          # ModelTrainer — fit loop + callbacks
│   ├── generator.py        # TextGenerator — temperature sampling
│   └── experiments.py      # Named architecture presets (bonus task)
├── data/shakespeare.txt    # Dataset (see "Dataset" below)
├── models/                 # Saved models, vocab, training history
└── outputs/                # Generated text samples (JSON)
```

### OOP / design highlights
- **Polymorphism** — `Tokenizer` is an abstract base; `CharTokenizer` and
  `WordTokenizer` are interchangeable, so the pipeline never branches on mode.
- **Factory** — `build_tokenizer(mode)` selects the implementation.
- **Facade** — `Corpus` hides cleaning → tokenising → windowing behind one call.
- **Encapsulation** — `LSTMTextModel` wraps build/compile/save/load; callers
  never touch raw Keras.
- **Single config tree** — `ExperimentConfig` (frozen dataclasses) makes every
  hyper-parameter explicit and experiments reproducible.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install tensorflow-cpu numpy
```

## Dataset

Shakespeare's Complete Works from Project Gutenberg (public domain):

```bash
mkdir -p data
curl -o data/shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

Link: https://www.gutenberg.org/ebooks/100

## Usage

```bash
# Train the baseline experiment (char-level LSTM)
.venv/bin/python main.py train

# Train a different architecture (bonus task)
.venv/bin/python main.py train --experiment deep_lstm

# Generate text from seeds (uses the trained checkpoint)
.venv/bin/python main.py generate --seed "to be or not to be" --seed "shall i compare thee"
```

## Pipeline stages

1. **Preprocessing** — lowercase, strip punctuation, collapse whitespace,
   tokenise (char/word), build sliding-window `(sequence → next-token)` pairs,
   split into train/validation.
2. **Model** — `Embedding → stacked LSTM → Dropout → Dense(softmax)`, compiled
   with `sparse_categorical_crossentropy` + Adam.
3. **Training** — `EarlyStopping` (restores best weights), `ModelCheckpoint`
   (best val_loss), and `ReduceLROnPlateau`. History saved to JSON.
4. **Generation** — autoregressive sampling with a **temperature** knob
   (low = conservative, high = diverse).

## Bonus: architecture experiments

Presets in `src/experiments.py` let you compare architectures on equal footing:

| Experiment   | Tokeniser | Seq len | LSTM layers   | Notes |
|--------------|-----------|---------|---------------|-------|
| `baseline`   | char      | 80      | (256,)        | Reference model |
| `deep_lstm`  | char      | 100     | (256, 256)    | Deeper — more long-range capacity |
| `short_seq`  | char      | 40      | (128,)        | Shorter context, faster, less coherent |
| `word_level` | word      | 20      | (256,)        | Generates whole words |

See `EXPERIMENT_REPORT.md` for the comparison write-up and sample outputs.
