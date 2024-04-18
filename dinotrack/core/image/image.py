from pathlib import Path
from typing import Union
from PIL import Image

from transformers import AutoImageProcessor
import numpy as np

from dinotrack.core.image.config import ReadImageConfig


ImageInput = Union[str, Image.Image, np.ndarray]


class ReadImage:
    def __init__(self, config: dict = {}):
        self.config = config = ReadImageConfig(**config)
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.processor.crop_size = {"width": config.width, "height": config.height}

    def __call__(self, image: Union[ImageInput, list[ImageInput]]):
        try:
            image = [Image.open(i) for i in image]
        except:
            if isinstance(image, str) or isinstance(image, Path):
                image = Image.open(image)

        return self.processor(image, **self.config.kwargs)
