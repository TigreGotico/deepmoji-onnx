"""Basic DeepMoji inference — auto-downloads model on first run."""

from deepmoji_onnx import DeepMojiONNX

dm = DeepMojiONNX.from_pretrained()  # downloads fp32 to ~/.cache/deepmoji/

sentences = [
    "I love this so much!!!",
    "This is absolutely terrible",
    "Having a great day :)",
    "I hate Mondays",
    "So proud of you ❤️",
]

print("=== Top-5 emoji per sentence ===")
for sent, emojis in zip(sentences, dm.top_emojis(sentences, k=5)):
    ranked = sorted(emojis.items(), key=lambda x: x[1], reverse=True)
    bar = "  ".join(f"{e} {p:.0%}" for e, p in ranked)
    print(f"  {sent!r:40s}  {bar}")
