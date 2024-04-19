from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import transformers
from PIL import Image
from transformers import AutoImageProcessor

from dinotrack.settings import DEFAULT_HEIGHT, DEFAULT_MODEL, DEFAULT_WIDTH


@dataclass
class ReadImageConfig:
    """
    Configuration class for reading images.

    Attributes:
        width (int): The width of the image. Defaults to DEFAULT_WIDTH.
        height (int): The height of the image. Defaults to DEFAULT_HEIGHT.
        model_name (str): The name of the model. Defaults to DEFAULT_MODEL.
        kwargs (dict): Additional keyword arguments. Defaults to an empty dictionary.

    Methods:
        __post_init__(): Initializes the object and sets default values for missing attributes.
    """

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    model_name: str = DEFAULT_MODEL
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Initializes the object and sets default values for missing attributes.
        If "return_tensors" is not present in kwargs, it is added with the value "pt".
        """
        if "return_tensors" not in self.kwargs:
            self.kwargs["return_tensors"] = "pt"


class ReadImage:
    """
    Class to read and process images using a pre-trained image processor.

    Args:
        config (dict): Configuration parameters for the image processor.

    Attributes:
        config (ReadImageConfig): Configuration object for the image processor.
        processor (AutoImageProcessor): Pre-trained image processor.
    """

    ImageInput = Union[str, Image.Image, np.ndarray]

    def __init__(self, config: dict = {}) -> None:
        self.config = config = ReadImageConfig(**config)
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.processor.crop_size = {"width": config.width, "height": config.height}

    def __call__(
        self, image: Union[ImageInput, list[ImageInput]]
    ) -> transformers.image_processing_utils.BatchFeature:
        """
        Process the input image(s) using the pre-trained image processor.

        Args:
            image (Union[ImageInput, list[ImageInput]]): Input image(s) to be processed.

        Returns:
            Processed image(s) based on the configuration parameters.
        """
        try:
            image = [Image.open(i) for i in image]
        except:
            if isinstance(image, str) or isinstance(image, Path):
                image = Image.open(image)

        return self.processor(image, **self.config.kwargs)
