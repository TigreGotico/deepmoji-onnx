# Model Variants & Quantization

## Comparison

| Variant | File | Size | Latency¹ | Top-1 match vs fp32 | Notes |
|---------|------|------|----------|--------------------|-------|
| `fp32`  | `deepmoji_fp32.onnx` | 90 MB | ~10 ms | — (reference) | Unrolled LSTM (241 Gemm). |
| `fp16`  | `deepmoji_fp16.onnx` | 46 MB | ~12 ms | 100% | Same unrolled graph, half weights. |
| `int8`  | `deepmoji_int8.onnx` | 51 MB | ~16 ms | ~90% | Loop-based LSTM + dynamic quant. |

¹ Batch of 6, CPU.

## Bundling

At 46-90 MB the models are too large to bundle inside a Python wheel (PyPI recommends
under 15 MB per file). The package downloads the selected variant on first use and caches it
in `~/.cache/deepmoji/`. See [`inference.py:DeepMojiONNX.from_pretrained`](api.md).

## Why int8 differs

The int8 model uses the scripted (Loop-op) export instead of the traced (unrolled) export.
The Loop op runs the LSTM cell in a different numerical order at the hardware level. This
causes small floating-point differences that can change the top-1 prediction for close
races. The distribution and ranking of the top-5 emoji stay consistent.

## fp16 vs fp32

fp16 gives the same top-1 predictions as fp32 on CPU with onnxruntime. The half-precision
rounding stays below the softmax discrimination threshold for all tested inputs.

## Quantization pipeline

```
pytorch_model.bin
    └─ DeepMojiONNXModel (fp32 torch) ──trace──► deepmoji_fp32.onnx   (90 MB)
    └─ DeepMojiONNXModel (fp16 torch) ──trace──► deepmoji_fp16.onnx   (46 MB)
    └─ DeepMojiONNXModel (fp32 torch) ──script─► Loop ONNX
                                       ──quantize_dynamic──► deepmoji_int8.onnx  (51 MB)
```

---
[← Export guide](export.md) · [Home](index.md) · [API reference →](api.md)
