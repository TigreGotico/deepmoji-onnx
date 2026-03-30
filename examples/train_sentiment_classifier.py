"""Train a sentiment classifier on IMDB using DeepMoji emoji features.

DeepMoji produces a 64-d softmax over emoji categories.  These probabilities
capture emotional tone well enough to train a strong linear classifier with
very little data — no fine-tuning required.

Dependencies:
    pip install deepmoji-onnx datasets scikit-learn

Usage:
    python train_sentiment_classifier.py          # default: 2 000 train, 500 test
    python train_sentiment_classifier.py --n 5000 --test 1000

What happens:
    1. Load IMDB reviews from HuggingFace datasets (auto-cached).
    2. Extract 64-d emoji feature vectors with DeepMojiONNX.
    3. Train a Logistic Regression on the features.
    4. Report accuracy, classification report, and a few live predictions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from deepmoji_onnx import DeepMojiONNX

BATCH_SIZE = 64
LABELS = ["negative", "positive"]


def extract_features(model: DeepMojiONNX, texts: list[str]) -> np.ndarray:
    """Batch-encode texts → (N, 64) emoji feature matrix."""
    features = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        features.append(model.predict(batch))
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  encoded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(features)


def main(n_train: int = 2000, n_test: int = 500, variant: str = "fp16") -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading IMDB …")
    ds = load_dataset("imdb", split={"train": "train", "test": "test"})
    rng = np.random.default_rng(42)

    def sample(split, n: int) -> tuple[list[str], np.ndarray]:
        idx = rng.choice(len(ds[split]), size=min(n, len(ds[split])), replace=False)
        texts = [ds[split][int(i)]["text"][:512] for i in idx]   # cap at 512 chars
        labels = np.array([ds[split][int(i)]["label"] for i in idx])
        return texts, labels

    train_texts, y_train = sample("train", n_train)
    test_texts,  y_test  = sample("test",  n_test)
    print(f"  train={len(train_texts)}  test={len(test_texts)}")

    # ------------------------------------------------------------------
    # 2. Extract DeepMoji features
    # ------------------------------------------------------------------
    print(f"\nLoading DeepMoji ({variant}) …")
    dm = DeepMojiONNX.from_pretrained(variant=variant)

    print("Encoding train set …")
    t0 = time.perf_counter()
    X_train = extract_features(dm, train_texts)
    train_ms = (time.perf_counter() - t0) * 1000
    print(f"  {train_ms:.0f} ms  ({train_ms/len(train_texts):.1f} ms/sample)")

    print("Encoding test set …")
    t0 = time.perf_counter()
    X_test = extract_features(dm, test_texts)
    test_ms = (time.perf_counter() - t0) * 1000
    print(f"  {test_ms:.0f} ms  ({test_ms/len(test_texts):.1f} ms/sample)")

    # ------------------------------------------------------------------
    # 3. Train classifier
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression …")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train_s, y_train)
    print(f"  fit in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test_s)
    acc = (y_pred == y_test).mean()
    print(f"\nAccuracy: {acc:.3f}  ({acc*100:.1f}%)")
    print("\n" + classification_report(y_test, y_pred, target_names=LABELS))

    # ------------------------------------------------------------------
    # 5. Live predictions
    # ------------------------------------------------------------------
    demos = [
        "Absolutely brilliant film. One of the best I've seen in years.",
        "Complete waste of time. Terrible acting and a nonsensical plot.",
        "It was okay, nothing special but not bad either.",
        "I laughed, I cried, I loved every minute of it!",
        "Boring and predictable. I fell asleep halfway through.",
    ]
    print("Live predictions:")
    feats = extract_features(dm, demos)
    feats_s = scaler.transform(feats)
    preds = clf.predict(feats_s)
    probs = clf.predict_proba(feats_s)
    for text, pred, prob in zip(demos, preds, probs):
        label = LABELS[pred]
        conf = prob[pred]
        emoji_top = dm.top_emojis([text], k=1)[0]
        top_emoji = next(iter(emoji_top))
        print(f"  [{label:8s} {conf:.0%}] {top_emoji}  {text[:60]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=2000, help="Training samples")
    ap.add_argument("--test",    type=int, default=500,  help="Test samples")
    ap.add_argument("--variant", default="fp16", choices=["fp32", "fp16", "int8"])
    args = ap.parse_args()
    main(args.n, args.test, args.variant)
