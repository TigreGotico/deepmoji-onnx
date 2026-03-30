"""Use the 2304-d attention features for embedding / similarity tasks.

Requires a feature-mode ONNX export:
    deepmoji-export --weights pytorch_model.bin --vocab vocabulary.json \\
        --out deepmoji_feat.onnx --mode feature
"""

from __future__ import annotations

import numpy as np

from deepmoji_onnx import DeepMojiONNX

# Load a feature-mode model (exported with --mode feature)
dm = DeepMojiONNX.from_pretrained(variant="fp32", mode="feature")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


sentences = [
    "I'm so happy today!",
    "What a wonderful day",
    "I feel great",
    "This is terrible",
    "I'm devastated",
]

feats = dm.encode(sentences)  # (N, 2304)

print("=== Pairwise emotional similarity ===")
ref = 0  # compare everything to sentences[0]
for i, sent in enumerate(sentences):
    sim = cosine_similarity(feats[ref], feats[i])
    print(f"  {sentences[ref]!r:25s}  vs  {sent!r:25s}  cos={sim:.3f}")
