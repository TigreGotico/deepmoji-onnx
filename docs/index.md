# deepmoji-onnx

ONNX inference library for DeepMoji (torchMoji). Runtime requires only `onnxruntime` and `numpy`.

## Architecture

```
tokens (B, T) int64
    └─ Embedding(50000, 256)         embed.weight  [50 000 × 256 fp32]
    └─ tanh
    └─ BiLSTM layer 0                lstm_0.*      [8 weight tensors]
       input=256, hidden=512, bi → output (B, T, 1024)
    └─ BiLSTM layer 1                lstm_1.*      [8 weight tensors]
       input=1024, hidden=512, bi → output (B, T, 1024)
    └─ Skip-concat [lstm_1 | lstm_0 | embed]       (B, T, 2304)
    └─ Self-attention pool           attention_layer.attention_vector [2304]
       → weighted representation     (B, 2304)
    └─ [emoji mode]  Linear(2304→64) + Softmax     → (B, 64)
       [feature mode] identity                     → (B, 2304)
```

**LSTM gates** use hard sigmoid: `clamp(0.2x + 0.5, 0, 1)` (matches original Keras/torchMoji).

**Attention mask**: pad positions (token == 0) receive zero attention weight so they
do not affect the output representation, even without `pack_padded_sequence`.

## Files

| File | Description |
|------|-------------|
| `export.py` | PyTorch → ONNX export (fp32/fp16/int8) |
| `deepmoji_onnx/inference.py` | `DeepMojiONNX` — `DeepMojiONNX.predict`, `.top_emojis`, `.encode` — `inference.py:35` |
| `deepmoji_onnx/tokenizer.py` | `DeepMojiTokenizer` — `tokenizer.py:43` |

## Navigation

- [Export guide](export.md)
- [Model variants & quantization](quantization.md)
- [API reference](api.md)
- [HuggingFace model repo](https://huggingface.co/OpenVoiceOS/deepmoji-onnx) *(link once uploaded)*
