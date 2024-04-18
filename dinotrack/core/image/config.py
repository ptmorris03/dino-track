from dataclasses import dataclass, field
from dinotrack.settings import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MODEL


@dataclass
class ReadImageConfig:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    model_name: str = DEFAULT_MODEL
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if "return_tensors" not in self.kwargs:
            self.kwargs["return_tensors"] = "pt"
