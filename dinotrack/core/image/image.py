from pathlib import Path
from typing import Union
from PIL import Image

import numpy as np
import transformers
from transformers import AutoImageProcessor

from dinotrack.core.image.config import ReadImageConfig


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
