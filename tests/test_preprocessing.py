"""Unit tests for the deterministic preprocessing components.

Run with:  .venv/bin/python -m pytest tests/ -q
(These tests need only numpy, not TensorFlow.)
"""

import numpy as np

from src.preprocessing import (
    CharTokenizer,
    SequenceBuilder,
    TextCleaner,
    WordTokenizer,
    build_tokenizer,
)


def test_cleaner_lowercases_and_strips_punctuation():
    assert TextCleaner.clean("Hello, WORLD!!  It's   me.") == "hello world it s me"


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer().fit("abcabc")
    assert tok.vocab_size == 3
    ids = tok.encode("cab")
    assert tok.decode(list(ids)) == "cab"


def test_word_tokenizer_roundtrip():
    tok = WordTokenizer().fit("the cat sat on the mat")
    assert tok.vocab_size == 5  # the, cat, sat, on, mat
    assert tok.decode(list(tok.encode("the cat"))) == "the cat"


def test_factory_returns_correct_type():
    assert isinstance(build_tokenizer("char"), CharTokenizer)
    assert isinstance(build_tokenizer("word"), WordTokenizer)


def test_factory_rejects_unknown_mode():
    try:
        build_tokenizer("bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown mode")


def test_sequence_builder_shapes_and_targets():
    encoded = np.arange(20, dtype=np.int32)
    builder = SequenceBuilder(sequence_length=5, step=1, validation_split=0.0)
    X, y = builder.build(encoded)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    # First window is [0,1,2,3,4] -> target 5.
    assert list(X[0]) == [0, 1, 2, 3, 4]
    assert y[0] == 5


def test_train_val_split_proportions():
    X = np.zeros((100, 5)); y = np.zeros(100)
    builder = SequenceBuilder(5, 1, validation_split=0.2)
    Xtr, ytr, Xv, yv = builder.train_val_split(X, y)
    assert len(Xv) == 20 and len(Xtr) == 80


def test_sequence_builder_raises_on_short_corpus():
    builder = SequenceBuilder(sequence_length=50, step=1, validation_split=0.0)
    try:
        builder.build(np.arange(10, dtype=np.int32))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for too-short corpus")
