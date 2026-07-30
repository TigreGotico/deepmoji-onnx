# deepmoji-onnx

ONNX inference library for [DeepMoji](https://github.com/bfelbo/DeepMoji) / [torchMoji](https://github.com/huggingface/torchMoji).

Runtime dependencies are `onnxruntime` and `numpy` only. The library does not need PyTorch.

## Quick start

```bash
pip install deepmoji-onnx
```

```python
from deepmoji_onnx import DeepMojiONNX

# Downloads model on first use (~90 MB fp32 / ~45 MB fp16 / ~51 MB int8)
dm = DeepMojiONNX.from_pretrained()                  # fp32 default
dm = DeepMojiONNX.from_pretrained(variant="fp16")
dm = DeepMojiONNX.from_pretrained(variant="int8")

# Top-k emoji per sentence
dm.top_emojis(["I love this so much!!!", "This is absolutely terrible"], k=5)
# [{'❤': 0.17, '💙': 0.10, ...}, {'😡': 0.12, ...}]

# Raw (N, 64) softmax probabilities
probs = dm.predict(sentences)

# (N, 2304) attention feature vectors  (use variant="feature" export)
feats = dm.encode(sentences)
```

## Model variants

| Variant | Size  | Latency¹ | Notes |
|---------|-------|----------|-------|
| `fp32`  | 90 MB | ~10 ms   | Default, unrolled LSTM, fastest runtime. |
| `fp16`  | 46 MB | ~12 ms   | Half-precision weights, about 2x smaller. |
| `int8`  | 51 MB | ~16 ms   | Int8-quantized Loop-op LSTM, predictions shift slightly vs fp32. |

¹ Batch of 6 sentences, CPU, AMD Ryzen 9 5950X.

## Exporting from source weights

If you have the original torchMoji `pytorch_model.bin`:

```bash
pip install deepmoji-onnx[export]   # adds torch, onnx

# fp32 (default)
deepmoji-export --weights pytorch_model.bin --vocab vocabulary.json --out deepmoji_fp32.onnx

# fp16
deepmoji-export --weights pytorch_model.bin --vocab vocabulary.json --out deepmoji_fp16.onnx --dtype fp16

# int8  (uses scripted Loop-based ONNX, then quantizes)
deepmoji-export --weights pytorch_model.bin --vocab vocabulary.json --out deepmoji_int8.onnx --dtype int8
```

See [`docs/export.md`](docs/export.md) for the full option list.

## Architecture

DeepMoji is a 2-layer bidirectional LSTM with self-attention, trained on 1.2 B tweets
to predict 64 emoji classes. The attention output (2304-d) is a general-purpose
emotional sentence representation you can use for transfer learning.

```
text → tokenize → Embedding(50k, 256) → BiLSTM×2 → Attention → [Softmax(64) | 2304-d]
```

This ONNX port rewrites the hard-sigmoid LSTM gates as `clamp(0.2x+0.5, 0, 1)`. The
result is fully ONNX-compatible and does not need `pack_padded_sequence`. Batch size
and sequence length are dynamic axes.

See [`docs/index.md`](docs/index.md) for the full architecture notes.

## Related projects

- [TigreGotico/emotion-algebra](https://github.com/TigreGotico/emotion-algebra) combines lexicon-based emotion scoring with this library's neural emoji-sentiment model through its `DeepMojiONNXAdapter`.

## License

Apache 2.0. Original DeepMoji weights: MIT (bfelbo/DeepMoji).
