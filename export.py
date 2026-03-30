"""Export DeepMoji (torchMoji) to ONNX.

Usage:
    uv run python export.py \
        --weights /path/to/pytorch_model.bin \
        --vocab   /path/to/vocabulary.json \
        --out     deepmoji.onnx \
        [--mode   emoji|feature]   # default: emoji (64-class softmax)
        [--maxlen 30]

The script loads the pretrained torchMoji weights, wraps the model in an
ONNX-compatible module (no pack_padded_sequence, hard-sigmoid implemented via
clamp), and exports with dynamic batch/sequence-length axes.

Dependencies (export only):
    torch>=2.0, onnx, onnxruntime (for verify)

The resulting .onnx file has NO PyTorch dependency at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hard sigmoid  (matches torchMoji's original definition exactly)
# ---------------------------------------------------------------------------

def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Clamp-based hard sigmoid: clamp(0.2x + 0.5, 0, 1)."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ONNX-compatible BiLSTM with hard-sigmoid gates
# ---------------------------------------------------------------------------

class _HardSigBiLSTM(nn.Module):
    """Single bidirectional LSTM layer using hard-sigmoid gates.

    Weight parameter names intentionally match those produced by
    ``LSTMHardSigmoid`` so we can load straight from the pretrained state dict.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        H, I = hidden_size, input_size
        # forward
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * H, I))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih_l0 = nn.Parameter(torch.zeros(4 * H))
        self.bias_hh_l0 = nn.Parameter(torch.zeros(4 * H))
        # reverse
        self.weight_ih_l0_reverse = nn.Parameter(torch.empty(4 * H, I))
        self.weight_hh_l0_reverse = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih_l0_reverse = nn.Parameter(torch.zeros(4 * H))
        self.bias_hh_l0_reverse = nn.Parameter(torch.zeros(4 * H))
        self.hidden_size = H

    def _cell(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = F.linear(x, w_ih, b_ih) + F.linear(h, w_hh, b_hh)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = _hard_sigmoid(i)
        f = _hard_sigmoid(f)
        g = torch.tanh(g)
        o = _hard_sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run over (batch, seq, input_size), return (batch, seq, 2*hidden)."""
        B, T, _ = x.shape
        H = self.hidden_size
        dev = x.device

        h_f = torch.zeros(B, H, device=dev, dtype=x.dtype)
        c_f = torch.zeros(B, H, device=dev, dtype=x.dtype)
        h_r = torch.zeros(B, H, device=dev, dtype=x.dtype)
        c_r = torch.zeros(B, H, device=dev, dtype=x.dtype)

        fwd_out = []
        rev_out = []
        for t in range(T):
            h_f, c_f = self._cell(
                x[:, t, :], h_f, c_f,
                self.weight_ih_l0, self.weight_hh_l0,
                self.bias_ih_l0, self.bias_hh_l0,
            )
            fwd_out.append(h_f.unsqueeze(1))

        for t in range(T - 1, -1, -1):
            h_r, c_r = self._cell(
                x[:, t, :], h_r, c_r,
                self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse, self.bias_hh_l0_reverse,
            )
            rev_out.append(h_r.unsqueeze(1))
        rev_out.reverse()

        fwd = torch.cat(fwd_out, dim=1)   # (B, T, H)
        rev = torch.cat(rev_out, dim=1)   # (B, T, H)
        return torch.cat([fwd, rev], dim=-1)  # (B, T, 2H)


# ---------------------------------------------------------------------------
# Attention layer (ONNX-compatible rewrite)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    """Self-attention pooling over time; produces fixed-size representation."""

    def __init__(self, attention_size: int) -> None:
        super().__init__()
        self.attention_vector = nn.Parameter(torch.empty(attention_size))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, attention_size)
            lengths: (B,) actual sequence lengths (int64)

        Returns:
            (B, attention_size) weighted representation
        """
        logits = x.matmul(self.attention_vector)          # (B, T)
        unnorm = (logits - logits.max(dim=1, keepdim=True).values).exp()

        T = x.size(1)
        idx = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        mask = (idx < lengths.unsqueeze(1)).to(x.dtype)         # (B, T)

        masked = unnorm * mask
        att = masked / (masked.sum(dim=1, keepdim=True) + 1e-9)  # (B, T)
        weighted = (x * att.unsqueeze(-1)).sum(dim=1)             # (B, attn_size)
        return weighted


# ---------------------------------------------------------------------------
# Full ONNX-export wrapper
# ---------------------------------------------------------------------------

class DeepMojiONNXModel(nn.Module):
    """Drop-in ONNX-exportable DeepMoji.

    Inputs:
        tokens:  (B, T) int64 — token ids, 0 = padding
        lengths: (B,)   int64 — actual sequence lengths (must be ≥ 1)

    Outputs (depending on mode):
        'emoji'   → (B, 64)   float32  emoji softmax probabilities
        'feature' → (B, 2304) float32  penultimate attention output
    """

    def __init__(self, mode: str = "emoji") -> None:
        super().__init__()
        assert mode in ("emoji", "feature"), "mode must be 'emoji' or 'feature'"
        self.mode = mode

        emb_dim = 256
        h = 512
        attn_size = 4 * h + emb_dim  # 2304

        self.embed = nn.Embedding(50000, emb_dim)
        self.lstm_0 = _HardSigBiLSTM(emb_dim, h)
        self.lstm_1 = _HardSigBiLSTM(h * 2, h)
        self.attention_layer = _Attention(attn_size)

        if mode == "emoji":
            self.output_layer = nn.Sequential(
                nn.Linear(attn_size, 64),
                nn.Softmax(dim=-1),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.tanh(self.embed(tokens))      # (B, T, 256)
        h0 = self.lstm_0(x)                     # (B, T, 1024)
        h1 = self.lstm_1(h0)                    # (B, T, 1024)

        # skip-connection: concat lstm_1 | lstm_0 | embed  → (B, T, 2304)
        combined = torch.cat([h1, h0, x], dim=-1)

        rep = self.attention_layer(combined, lengths)  # (B, 2304)

        if self.mode == "emoji":
            return self.output_layer(rep)              # (B, 64)
        return rep                                     # (B, 2304)


# ---------------------------------------------------------------------------
# Weight loading from the original torchMoji state dict
# ---------------------------------------------------------------------------

def _load_torchmoji_weights(
    export_model: DeepMojiONNXModel,
    weight_path: str,
    mode: str,
) -> None:
    """Map original LSTMHardSigmoid parameter names → _HardSigBiLSTM names."""
    src: dict[str, torch.Tensor] = torch.load(weight_path, map_location="cpu")

    remap = {
        # embed
        "embed.weight": "embed.weight",
        # lstm_0  (LSTMHardSigmoid stores weights under lstm_0.*)
        "lstm_0.weight_ih_l0":         "lstm_0.weight_ih_l0",
        "lstm_0.weight_hh_l0":         "lstm_0.weight_hh_l0",
        "lstm_0.bias_ih_l0":           "lstm_0.bias_ih_l0",
        "lstm_0.bias_hh_l0":           "lstm_0.bias_hh_l0",
        "lstm_0.weight_ih_l0_reverse": "lstm_0.weight_ih_l0_reverse",
        "lstm_0.weight_hh_l0_reverse": "lstm_0.weight_hh_l0_reverse",
        "lstm_0.bias_ih_l0_reverse":   "lstm_0.bias_ih_l0_reverse",
        "lstm_0.bias_hh_l0_reverse":   "lstm_0.bias_hh_l0_reverse",
        # lstm_1
        "lstm_1.weight_ih_l0":         "lstm_1.weight_ih_l0",
        "lstm_1.weight_hh_l0":         "lstm_1.weight_hh_l0",
        "lstm_1.bias_ih_l0":           "lstm_1.bias_ih_l0",
        "lstm_1.bias_hh_l0":           "lstm_1.bias_hh_l0",
        "lstm_1.weight_ih_l0_reverse": "lstm_1.weight_ih_l0_reverse",
        "lstm_1.weight_hh_l0_reverse": "lstm_1.weight_hh_l0_reverse",
        "lstm_1.bias_ih_l0_reverse":   "lstm_1.bias_ih_l0_reverse",
        "lstm_1.bias_hh_l0_reverse":   "lstm_1.bias_hh_l0_reverse",
        # attention
        "attention_layer.attention_vector": "attention_layer.attention_vector",
    }
    if mode == "emoji":
        remap["output_layer.0.weight"] = "output_layer.0.weight"
        remap["output_layer.0.bias"]   = "output_layer.0.bias"

    dst_sd = export_model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    missing = []

    for src_k, dst_k in remap.items():
        if src_k in src:
            mapped[dst_k] = src[src_k]
        else:
            missing.append(src_k)

    if missing:
        print(f"[warn] Keys not found in checkpoint: {missing}", file=sys.stderr)

    dst_sd.update(mapped)
    export_model.load_state_dict(dst_sd)
    print(f"Loaded {len(mapped)} weight tensors from {weight_path}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_int8(model: "DeepMojiONNXModel", out_path: Path, opset: int) -> None:
    """Export via scripted (Loop-op) ONNX then apply dynamic int8 quantization."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
    import tempfile

    sm = torch.jit.script(model)
    tokens = torch.zeros(2, 30, dtype=torch.long)
    tokens[:, :5] = 1
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loop_path = out_path.with_suffix(".loop_tmp.onnx")
    pre_path = out_path.with_suffix(".pre_tmp.onnx")

    with torch.no_grad():
        torch.onnx.export(
            sm,
            (tokens, lengths),
            str(loop_path),
            input_names=["tokens", "lengths"],
            output_names=["output"],
            dynamic_axes={
                "tokens":  {0: "batch", 1: "seq_len"},
                "lengths": {0: "batch"},
                "output":  {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    shape_inference.quant_pre_process(str(loop_path), str(pre_path), skip_symbolic_shape=True)
    quantize_dynamic(
        str(pre_path),
        str(out_path),
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    loop_path.unlink(missing_ok=True)
    pre_path.unlink(missing_ok=True)


def export(
    weights: str,
    vocab: str,
    out: str,
    mode: str = "emoji",
    dtype: str = "fp32",
    maxlen: int = 30,
    opset: int = 17,
) -> None:
    """Export the model and write the .onnx file.

    Args:
        weights: Path to ``pytorch_model.bin``.
        vocab:   Path to ``vocabulary.json`` (copied alongside .onnx for inference).
        out:     Output path for the .onnx file.
        mode:    ``'emoji'`` (64-class softmax) or ``'feature'`` (2304-d embedding).
        dtype:   ``'fp32'``, ``'fp16'``, or ``'int8'``.
        maxlen:  Sequence length used for the dummy input.
        opset:   ONNX opset version.
    """
    assert dtype in ("fp32", "fp16", "int8"), "dtype must be fp32 | fp16 | int8"

    model = DeepMojiONNXModel(mode=mode)
    _load_torchmoji_weights(model, weights, mode)

    if dtype == "fp16":
        model = model.half()
    model.eval()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "int8":
        _export_int8(model, out_path, opset)
    else:
        B = 2
        dummy_tokens = torch.zeros(B, maxlen, dtype=torch.long)
        dummy_tokens[:, :5] = 1
        dummy_lengths = torch.tensor([5, 3], dtype=torch.long)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_tokens, dummy_lengths),
                str(out_path),
                input_names=["tokens", "lengths"],
                output_names=["output"],
                dynamic_axes={
                    "tokens":  {0: "batch", 1: "seq_len"},
                    "lengths": {0: "batch"},
                    "output":  {0: "batch"},
                },
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )

    print(f"Exported {dtype} ONNX model → {out_path}")

    # quick verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(out_path))
        vt = torch.zeros(2, maxlen, dtype=torch.long)
        vt[:, :5] = 1
        vl = torch.tensor([5, 3], dtype=torch.long)
        result = sess.run(None, {"tokens": vt.numpy(), "lengths": vl.numpy()})
        print(f"onnxruntime verify OK — output shape: {result[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping verify")

    # copy vocab next to the onnx file
    import shutil
    vocab_dst = out_path.with_name("vocabulary.json")
    if Path(vocab) != vocab_dst:
        shutil.copy(vocab, vocab_dst)
        print(f"Copied vocabulary → {vocab_dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DeepMoji to ONNX")
    ap.add_argument("--weights", required=True, help="Path to pytorch_model.bin")
    ap.add_argument("--vocab",   required=True, help="Path to vocabulary.json")
    ap.add_argument("--out",     default="deepmoji.onnx", help="Output .onnx path")
    ap.add_argument("--mode",    choices=["emoji", "feature"], default="emoji")
    ap.add_argument("--dtype",   choices=["fp32", "fp16", "int8"], default="fp32")
    ap.add_argument("--maxlen",  type=int, default=30)
    ap.add_argument("--opset",   type=int, default=17)
    args = ap.parse_args()
    export(args.weights, args.vocab, args.out, args.mode, args.dtype, args.maxlen, args.opset)


if __name__ == "__main__":
    main()
