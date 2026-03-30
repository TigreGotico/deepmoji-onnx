"""deepmoji_onnx — tiny inference library for DeepMoji ONNX models.

Runtime dependencies: onnxruntime, numpy (no PyTorch).
"""

from deepmoji_onnx.inference import DeepMojiONNX
from deepmoji_onnx.tokenizer import DeepMojiTokenizer

__all__ = ["DeepMojiONNX", "DeepMojiTokenizer"]
