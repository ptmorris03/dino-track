from dataclasses import dataclass, field

import transformers
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from dinotrack.settings import DEFAULT_HEIGHT, DEFAULT_MODEL, DEFAULT_WIDTH


@dataclass
class ModelConfig:
    """
    Configuration class for the model.

    Attributes:
        width (int): The width of the model.
        height (int): The height of the model.
        model_name (str): The name of the model.
        kwargs (dict): Additional keyword arguments for the model.
    """

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    model_name: str = DEFAULT_MODEL
    kwargs: dict = field(default_factory=dict)


class Model:
    """
    A class representing a model for image processing.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.

    Attributes:
        config (ModelConfig): An instance of the ModelConfig class containing the model configuration.
        model (AutoModel): The pretrained model used for image processing.

    """

    def __init__(self, config: dict = {}):
        self.config = config = ModelConfig(**config)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.crop_size = {"width": config.width, "height": config.height}

    def __call__(
        self, inputs: transformers.image_processing_utils.BatchFeature
    ) -> BaseModelOutputWithPooling:
        """
        Perform image processing using the model.

        Args:
            inputs (BatchFeature): The input batch of features for image processing.

        Returns:
            BaseModelOutputWithPooling: The output of the image processing, including the pooled features.

        """
        return self.model(**inputs, **self.config.kwargs)
