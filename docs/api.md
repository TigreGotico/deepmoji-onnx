# API Reference

## `DeepMojiONNX` (`deepmoji_onnx/inference.py:57`)

```python
class DeepMojiONNX:
    @classmethod
    def from_pretrained(
        cls,
        variant: str = "fp32",        # "fp32" | "fp16" | "int8"
        cache_dir: str | None = None, # default ~/.cache/deepmoji/
        maxlen: int = 30,
        providers: list[str] | None = None,
    ) -> "DeepMojiONNX": ...

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        emoji_path: str | Path | None = None,
        maxlen: int = 30,
        providers: list[str] | None = None,
    ) -> None: ...

    def predict(self, sentences: list[str]) -> np.ndarray:
        """(N, 64) float32 emoji softmax probabilities."""

    def top_emojis(self, sentences: list[str], k: int = 5) -> list[dict[str, float]]:
        """Top-k emoji per sentence. Returns emoji char if emoji_path was given, else index string."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        """(N, 2304) float32 attention feature vectors. Same as predict() for emoji-mode models.
        Use a feature-mode export for proper 2304-d embeddings."""
```

## `DeepMojiTokenizer` (`deepmoji_onnx/tokenizer.py:43`)

```python
class DeepMojiTokenizer:
    def __init__(
        self,
        vocab_path: str | Path,   # vocabulary.json
        maxlen: int = 30,
        mask_value: int = 0,
        unk_value: int = 1,
    ) -> None: ...

    def tokenize(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            tokens:  int64 (N, maxlen), zero-padded token ids
            lengths: int64 (N,),        actual sequence lengths (at least 1)
        """
```

### Tokenizer preprocessing

The tokenizer applies these steps in order: URL → `CUSTOM_URL`, @mention → `CUSTOM_AT`, number → `CUSTOM_NUMBER`,
then lower-case, then vocabulary lookup (unknown → `CUSTOM_UNKNOWN` id=1).

## `export.export()` (`export.py:270`)

```python
def export(
    weights: str,         # path to pytorch_model.bin
    vocab: str,           # path to vocabulary.json
    out: str,             # output .onnx path
    mode: str = "emoji",  # "emoji" | "feature"
    dtype: str = "fp32",  # "fp32" | "fp16" | "int8"
    maxlen: int = 30,
    opset: int = 17,
) -> None: ...
```

---
[← Model variants & quantization](quantization.md) · [Home](index.md)
