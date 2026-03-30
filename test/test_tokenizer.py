"""Unit tests for DeepMojiTokenizer — no model required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepmoji_onnx.tokenizer import DeepMojiTokenizer, _preprocess


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_url_replaced():
    tokens = _preprocess("Check https://example.com out")
    assert "CUSTOM_URL" in tokens


def test_at_mention_replaced():
    tokens = _preprocess("Hello @world how are you")
    assert "CUSTOM_AT" in tokens


def test_number_replaced():
    tokens = _preprocess("I have 42 cats")
    assert "CUSTOM_NUMBER" in tokens


def test_lowercased():
    tokens = _preprocess("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


# ---------------------------------------------------------------------------
# Tokenizer tests (uses a minimal in-memory vocab)
# ---------------------------------------------------------------------------

@pytest.fixture()
def vocab_file(tmp_path: Path) -> Path:
    vocab = {
        "CUSTOM_MASK": 0,
        "CUSTOM_UNKNOWN": 1,
        "CUSTOM_AT": 2,
        "CUSTOM_URL": 3,
        "CUSTOM_NUMBER": 4,
        "CUSTOM_BREAK": 5,
        "hello": 10,
        "world": 11,
        "love": 12,
        "this": 13,
    }
    p = tmp_path / "vocabulary.json"
    p.write_text(json.dumps(vocab))
    return p


def test_basic_tokenize(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=10)
    tokens, lengths = tok.tokenize(["hello world"])
    assert tokens.shape == (1, 10)
    assert lengths[0] == 2
    assert tokens[0, 0] == 10
    assert tokens[0, 1] == 11
    assert tokens[0, 2] == 0  # padding


def test_unknown_word(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=5)
    tokens, lengths = tok.tokenize(["foobar"])
    assert tokens[0, 0] == 1  # CUSTOM_UNKNOWN


def test_truncation(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=2)
    tokens, lengths = tok.tokenize(["hello world love this"])
    assert tokens.shape == (1, 2)
    assert lengths[0] == 2


def test_batch_shape(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=5)
    tokens, lengths = tok.tokenize(["hello", "world love", "this"])
    assert tokens.shape == (3, 5)
    assert lengths.shape == (3,)
    assert lengths[1] == 2


def test_empty_sentence_length_at_least_one(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=5)
    _, lengths = tok.tokenize([""])
    assert lengths[0] >= 1


def test_dtype(vocab_file):
    tok = DeepMojiTokenizer(vocab_file, maxlen=5)
    tokens, lengths = tok.tokenize(["hello"])
    assert tokens.dtype == np.int64
    assert lengths.dtype == np.int64
