from dataclasses import dataclass, field

import transformers
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

    def __call__(self, inputs: transformers.image_processing_utils.BatchFeature):
        return self.model(**inputs, **self.config.kwargs)
