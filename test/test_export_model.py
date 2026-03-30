"""Unit tests for the export-side DeepMojiONNXModel (requires torch).

These tests do NOT require the pretrained weights — they use random weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

# export.py lives at repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from export import DeepMojiONNXModel


def _init_small(model: "DeepMojiONNXModel") -> "DeepMojiONNXModel":
    """Scale all parameters to ±0.01 so random-weight LSTM doesn't blow up."""
    import torch
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.data.uniform_(-0.01, 0.01)
    return model


@pytest.fixture()
def emoji_model():
    m = DeepMojiONNXModel(mode="emoji")
    _init_small(m)
    m.eval()
    return m


@pytest.fixture()
def feature_model():
    m = DeepMojiONNXModel(mode="feature")
    _init_small(m)
    m.eval()
    return m


def _dummy_batch(B: int = 2, T: int = 10):
    import torch
    tokens = torch.randint(1, 100, (B, T), dtype=torch.long)
    lengths = torch.full((B,), T, dtype=torch.long)
    if B > 1:
        lengths[1] = max(1, T // 2)
    return tokens, lengths


def test_emoji_output_shape(emoji_model):
    tokens, lengths = _dummy_batch(2, 10)
    with torch.no_grad():
        out = emoji_model(tokens, lengths)
    assert out.shape == (2, 64)


def test_emoji_output_sums_to_one(emoji_model):
    tokens, lengths = _dummy_batch(3, 8)
    with torch.no_grad():
        out = emoji_model(tokens, lengths)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_feature_output_shape(feature_model):
    tokens, lengths = _dummy_batch(2, 10)
    with torch.no_grad():
        out = feature_model(tokens, lengths)
    assert out.shape == (2, 2304)


def test_single_sample(emoji_model):
    tokens, lengths = _dummy_batch(1, 5)
    with torch.no_grad():
        out = emoji_model(tokens, lengths)
    assert out.shape == (1, 64)


def test_onnx_export(emoji_model, tmp_path):
    import torch
    out_path = tmp_path / "model.onnx"
    tokens, lengths = _dummy_batch(2, 10)
    with torch.no_grad():
        torch.onnx.export(
            emoji_model,
            (tokens, lengths),
            str(out_path),
            input_names=["tokens", "lengths"],
            output_names=["output"],
            dynamic_axes={
                "tokens":  {0: "batch", 1: "seq_len"},
                "lengths": {0: "batch"},
                "output":  {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_onnx_runtime_inference(emoji_model, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import torch, numpy as np

    out_path = tmp_path / "model.onnx"
    tokens, lengths = _dummy_batch(2, 10)
    with torch.no_grad():
        torch.onnx.export(
            emoji_model,
            (tokens, lengths),
            str(out_path),
            input_names=["tokens", "lengths"],
            output_names=["output"],
            dynamic_axes={
                "tokens":  {0: "batch", 1: "seq_len"},
                "lengths": {0: "batch"},
                "output":  {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    sess = ort.InferenceSession(str(out_path))
    result = sess.run(None, {
        "tokens":  tokens.numpy(),
        "lengths": lengths.numpy(),
    })
    assert result[0].shape == (2, 64)
    sums = result[0].sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-5)
