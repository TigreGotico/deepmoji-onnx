"""Batch inference benchmark — compares fp32, fp16, int8 variants."""

import time
from deepmoji_onnx import DeepMojiONNX

SENTENCES = [
    "I love this so much!!!",
    "This is absolutely terrible",
    "Having a great day :)",
    "I hate Mondays",
    "So proud of you",
    "crying myself to sleep",
] * 10  # 60 sentences

for variant in ("fp32", "fp16", "int8"):
    dm = DeepMojiONNX.from_pretrained(variant=variant)

    # warm-up
    dm.predict(SENTENCES[:2])

    t0 = time.perf_counter()
    probs = dm.predict(SENTENCES)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(
        f"variant={variant:4s}  "
        f"sentences={len(SENTENCES):3d}  "
        f"shape={probs.shape}  "
        f"time={elapsed_ms:.1f}ms  "
        f"({elapsed_ms/len(SENTENCES):.2f}ms/sentence)"
    )
