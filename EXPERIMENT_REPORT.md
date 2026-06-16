# Experiment Report — LSTM Text Generation

## Dataset
- **Source:** Shakespeare's Complete Works, Project Gutenberg eBook #100
  (public domain). Download: https://www.gutenberg.org/files/100/100-0.txt
- **Preprocessing:** lowercased, punctuation removed, whitespace collapsed.
- **Corpus slice used:** first 300,000 characters (kept training feasible on CPU).
- **Tokenisation:** character-level → vocabulary of 37 tokens.
- **Windowing:** sliding window of length 80, step 4 → ~70k (input, next-char)
  pairs, split 90/10 train/validation.

## Baseline model (trained)

| Setting | Value |
|---|---|
| Architecture | Embedding(37→64) → LSTM(256) → Dropout(0.2) → Dense(37, softmax) |
| Parameters | ~340k |
| Loss / optimiser | sparse categorical cross-entropy / Adam (lr 1e-3) |
| Batch size | 128 |
| Regularisation | Dropout 0.2, EarlyStopping(patience 3), ReduceLROnPlateau |

### Training outcome
- Trained for 16 epochs; **early stopping restored the best weights from epoch 13.**
- Best **val_loss ≈ 1.533**, **val_accuracy ≈ 53.5%** (next-char prediction).
- After epoch 13 train-loss kept falling while val-loss rose → classic
  overfitting, correctly caught by early stopping.

| Epoch | train_loss | val_loss | val_acc |
|------:|-----------:|---------:|--------:|
| 1  | 2.45 | 2.15 | 35.9% |
| 3  | 1.94 | 1.83 | 44.1% |
| 6  | 1.68 | 1.63 | 50.6% |
| 10 | 1.50 | 1.55 | 52.6% |
| 13 | 1.39 | **1.53** | **53.5%** |
| 16 | 1.28 | 1.54 | 53.0% (stopped) |

### Sample output (temperature 0.8, 400 chars)

**Seed:** `to be or not to be`
> to be or not to be to mone and particate my good for what i will not it taunte the arter comes me of first lord i lad what thou art friends for him who have with mark and gaage my preyens of be so dispores even i no pacolles have to the to see to let my dear fortune sir can this cume first lord i did of him for many he with to true this part creater days she stot in my most deceepater lies...

**Seed:** `now is the winter of our discontent`
> now is the winter of our discontent thy tongur trath of the evines a star you my self to countess so ll power not to my part that you scene is a made the know bertore this love you seatch the covery once the poor is than make the reasisgy helena are see all thy would i think i have make an more beauty s of in thought...

**Observations.** The char-level model learned:
- real English function words and Shakespearean register ("thy", "thou art",
  "my lord", "my dear fortune sir");
- play structure — speaker tags like "first lord", "second", "countess", and
  character names ("helena", "parolles", "bertore" ≈ Bertram from *All's Well*);
- mostly well-formed word shapes, with occasional invented words — expected at
  val_loss ≈ 1.5 for a single-layer char model.

(Full outputs for all four seeds in `outputs/samples_baseline.json`.)

## Bonus — architecture experiments

The presets in `src/experiments.py` let each variant be trained and compared on
equal footing via `python main.py train --experiment <name>`:

| Experiment | Tokeniser | Seq len | LSTM | Expected effect |
|---|---|---|---|---|
| `baseline` | char | 80 | (256,) | Reference (trained above) |
| `deep_lstm` | char | 100 | (256, 256) | More capacity for long-range style/structure; slower per epoch, higher overfit risk → needs more dropout (0.3 set) |
| `short_seq` | char | 40 | (128,) | Less context → weaker long-range coherence, but ~2× faster to train |
| `word_level` | word | 20 | (256,) | Generates whole real words (no spelling errors) but needs a far larger vocab/embedding and more data to be fluent |

### How architecture affects quality (analysis)
- **Sequence length** is the single biggest lever for *coherence*: the baseline's
  80-char window lets it carry clause-level context; `short_seq` (40) produces
  text that is locally word-like but loses the thread faster.
- **Depth** (`deep_lstm`, 2 stacked layers) increases representational capacity
  and tends to lower train-loss faster, but on a 300k-char corpus it overfits
  sooner — so it pairs higher dropout with early stopping.
- **Tokenisation granularity** changes the failure mode entirely: char-level
  occasionally misspells; word-level never misspells but can produce
  grammatically loose word salad until it has seen enough data.
- **Temperature** at generation trades safety for variety: ~0.7 gives the most
  readable Shakespeare; ≥1.0 becomes inventive but noisier.

> Note: only the `baseline` was trained to completion here (each CPU run is
> ~45 min). The other presets are ready to run with the single command above
> and will write their own `outputs/samples_<name>.json`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
curl -o data/shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
.venv/bin/python main.py train                       # baseline
.venv/bin/python main.py generate --seed "to be or not to be"
```
