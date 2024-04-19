import transformers
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from dinotrack.core.model.config import ModelConfig


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
