import torch


DEFAULT_MODEL = "facebook/dinov2-small"
DEFAULT_WIDTH = 1792
DEFAULT_HEIGHT = 1792
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
