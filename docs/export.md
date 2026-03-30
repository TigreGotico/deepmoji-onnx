# Export Guide

## Prerequisites

```bash
pip install deepmoji-onnx[export]   # adds torch>=2.0, onnx>=1.14
```

You need the original torchMoji pretrained weights (86 MB):

```bash
wget "https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=1" -O pytorch_model.bin
```

And `vocabulary.json` (from this repo or from `torchMoji/model/`).

## CLI

```bash
# fp32 (default, fastest runtime)
deepmoji-export \
  --weights pytorch_model.bin \
  --vocab   vocabulary.json \
  --out     deepmoji_fp32.onnx

# fp16 (~2× smaller, same accuracy)
deepmoji-export ... --dtype fp16 --out deepmoji_fp16.onnx

# int8 (~4× smaller, slight accuracy shift)
deepmoji-export ... --dtype int8 --out deepmoji_int8.onnx

# feature extractor (2304-d embedding output, no softmax)
deepmoji-export ... --mode feature --out deepmoji_feat.onnx

# all options
deepmoji-export --help
```

## Python API

```python
from export import export

export(
    weights="pytorch_model.bin",
    vocab="vocabulary.json",
    out="deepmoji.onnx",
    mode="emoji",    # "emoji" | "feature"
    dtype="fp32",    # "fp32" | "fp16" | "int8"
    maxlen=30,       # sequence truncation used for dummy input
    opset=17,
)
```

## Implementation notes

- **fp32**: traced (unrolled LSTM, ~241 Gemm nodes), fastest inference.
- **fp16**: same graph as fp32 with all weights/compute in float16. Input tokens remain int64.
- **int8**: scripted (Loop-op LSTM), then `quantize_dynamic` with `QuantType.QInt8`. Slightly different top-1 predictions vs fp32 due to Loop vs unrolled numerical differences.
- All exports use `dynamo=False` (legacy TorchScript exporter) because the dynamo exporter cannot trace Python `for` loops.
- Dynamic axes: `tokens` (batch, seq_len), `lengths` (batch), `output` (batch). Any sequence length and batch size at runtime.
