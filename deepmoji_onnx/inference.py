"""DeepMoji ONNX inference — no PyTorch required at runtime.

Minimal dependencies: onnxruntime, numpy.

Example::

    from deepmoji_onnx import DeepMojiONNX

    # auto-download from HuggingFace on first use
    dm = DeepMojiONNX.from_pretrained()                  # fp32
    dm = DeepMojiONNX.from_pretrained(variant="fp16")
    dm = DeepMojiONNX.from_pretrained(variant="int8")

    # emoji probabilities (N, 64)
    probs = dm.predict(["I love this so much!", "I'm devastated"])

    # top-k emoji codes
    top = dm.top_emojis(["I love this so much!"], k=5)
    # [{':hearts:': 0.17, ':blue_heart:': 0.10, ...}]

    # raw 2304-d feature vectors (for feature-mode model)
    feats = dm.encode(["text here"])
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Install onnxruntime: uv pip install onnxruntime") from e

from deepmoji_onnx.tokenizer import DeepMojiTokenizer


# ---------------------------------------------------------------------------
# HuggingFace repo coordinates
# ---------------------------------------------------------------------------

_HF_REPO = "TigreGotico/deepmoji-onnx"
_HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/main"

_VARIANT_FILES = {
    "fp32": "deepmoji_fp32.onnx",
    "fp16": "deepmoji_fp16.onnx",
    "int8": "deepmoji_int8.onnx",
}

_AUX_FILES = ["vocabulary.json", "emoji_codes.json"]

# SHA-256 digests for tamper detection
_CHECKSUMS: dict[str, Optional[str]] = {
    "deepmoji_fp32.onnx": "2aa7c51e93e70df7b84ad03fdbe969947fd223623c97346e8df633674368ac84",
    "deepmoji_fp16.onnx": "b1a20fe494d897de069d187937a86020d121bc2ba4d9f98561b85f9eb41846df",
    "deepmoji_int8.onnx": "41e1fc6fd00aa815395524f400065e4721bf627fbf7e2b331ccaf3c63c988760",
    "vocabulary.json":    "951e74fe4661b23e490e79d3a9496bd61d027b14b55dd99dd1ae2c1fc3e25f3e",
    "emoji_codes.json":   "716571ba93b97862ace5c9dfb05e0df7d888f5b0385d41237e272f95845b4e24",
}


def _default_cache_dir() -> Path:
    return Path(os.environ.get("DEEPMOJI_CACHE", Path.home() / ".cache" / "deepmoji"))


def _verify_sha256(path: Path, expected: Optional[str]) -> None:
    if expected is None:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: got {digest}, expected {expected}. "
            "File removed; re-run to re-download."
        )


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"{size_mb:.1f} MB")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_pretrained(
    variant: str = "fp32",
    cache_dir: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path, Path]:
    """Ensure model/vocab/emoji_codes are cached locally.

    Args:
        variant:   ``'fp32'`` | ``'fp16'`` | ``'int8'``.
        cache_dir: Where to store downloaded files (default ``~/.cache/deepmoji``).

    Returns:
        (model_path, vocab_path, emoji_path)
    """
    if variant not in _VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_VARIANT_FILES)}")

    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    model_file = _VARIANT_FILES[variant]
    to_fetch = [model_file] + _AUX_FILES

    for fname in to_fetch:
        dest = cache / fname
        if not dest.exists():
            _download(f"{_HF_BASE}/{fname}", dest)
            _verify_sha256(dest, _CHECKSUMS.get(fname))

    return cache / model_file, cache / "vocabulary.json", cache / "emoji_codes.json"


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class DeepMojiONNX:
    """Inference wrapper for an ONNX-exported DeepMoji model.

    Args:
        model_path:  Path to ``deepmoji_*.onnx``.
        vocab_path:  Path to ``vocabulary.json``.
        emoji_path:  Optional path to ``emoji_codes.json``. If given, ``top_emojis``
                     returns emoji chars; otherwise returns index strings.
        maxlen:      Sequence truncation length (must match export; default 30).
        providers:   onnxruntime execution providers (default: auto-select).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        vocab_path: Union[str, Path],
        emoji_path: Optional[Union[str, Path]] = None,
        maxlen: int = 30,
        providers: Optional[list[str]] = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers or ort.get_available_providers(),
        )
        self._tokenizer = DeepMojiTokenizer(vocab_path, maxlen=maxlen)

        out_shape = self._session.get_outputs()[0].shape
        self._is_feature_mode = (
            out_shape[-1] != 64 if isinstance(out_shape[-1], int) else False
        )

        self._emoji_codes: Optional[list[str]] = None
        if emoji_path is not None:
            emoji_path = Path(emoji_path)
            if emoji_path.exists():
                with emoji_path.open(encoding="utf-8") as f:
                    raw = json.load(f)
                n = len(raw)
                self._emoji_codes = [raw.get(str(i), str(i)) for i in range(n)]

    # ------------------------------------------------------------------
    # Class-level constructor with auto-download
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "fp32",
        cache_dir: Optional[Union[str, Path]] = None,
        maxlen: int = 30,
        providers: Optional[list[str]] = None,
    ) -> "DeepMojiONNX":
        """Load a pretrained model, downloading it from HuggingFace if not cached.

        Args:
            variant:   ``'fp32'`` (default) | ``'fp16'`` | ``'int8'``.
            cache_dir: Local cache directory (default ``$DEEPMOJI_CACHE`` or
                       ``~/.cache/deepmoji``).
            maxlen:    Sequence truncation length.
            providers: onnxruntime execution providers.

        Returns:
            Ready-to-use ``DeepMojiONNX`` instance.
        """
        model_path, vocab_path, emoji_path = download_pretrained(variant, cache_dir)
        return cls(model_path, vocab_path, emoji_path, maxlen=maxlen, providers=providers)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, sentences: list[str]) -> np.ndarray:
        """Run the model on ``sentences``.

        Returns:
            float32 array (N, 64) for emoji mode or (N, 2304) for feature mode.
        """
        tokens, lengths = self._tokenizer.tokenize(sentences)
        outputs = self._session.run(
            None,
            {"tokens": tokens, "lengths": lengths},
        )
        return outputs[0]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Return emotional feature vectors (2304-d for feature-mode exports,
        64-d softmax for emoji-mode exports).
        """
        return self.predict(sentences)

    def top_emojis(
        self,
        sentences: list[str],
        k: int = 5,
    ) -> list[dict[str, float]]:
        """Return top-k emoji predictions per sentence.

        Args:
            sentences: Input texts.
            k:         Number of top emojis.

        Returns:
            List of dicts: emoji char (or index string) → probability, sorted by prob.
        """
        probs = self.predict(sentences)
        results: list[dict[str, float]] = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][:k]
            if self._emoji_codes:
                entry = {self._emoji_codes[i]: float(row[i]) for i in top_idx}
            else:
                entry = {str(i): float(row[i]) for i in top_idx}
            results.append(entry)
        return results

    def __repr__(self) -> str:
        mode = "feature" if self._is_feature_mode else "emoji"
        return (
            f"DeepMojiONNX(mode={mode!r}, "
            f"tokenizer={self._tokenizer!r}, "
            f"providers={self._session.get_providers()!r})"
        )
