from pathlib import Path
from typing import Union
from PIL import Image
from transformers import AutoImageProcessor
from dataclasses import dataclass, field
import numpy as np
from transformers import AutoModel

from dinotrack.settings import DEFAULT_MODEL


@dataclass
class ModelConfig:
    model_name: str = DEFAULT_MODEL
    kwargs: dict = field(default_factory=dict)


class Model:
    def __init__(self, config: dict = {}):
        self.config = config = ModelConfig(**config)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.crop_size = {"width": config.width, "height": config.height}

    def __call__(self, image: Union[ImageInput, list[ImageInput]]):
        try:
            image = [Image.open(i) for i in image]
        except:
            if isinstance(image, str) or isinstance(image, Path):
                image = Image.open(image)

        return self.model(image, **self.config.kwargs)
