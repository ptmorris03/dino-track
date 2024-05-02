from collections import namedtuple
from dataclasses import dataclass, field
from typing import Union

import torch
import transformers
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from dinotrack.core.model.class_head import ClassHead, ClassHeadConfig
from dinotrack.core.model.model import ModelConfig
from dinotrack.settings import DEVICE, DTYPE


@dataclass
class TrackingModelConfig:
    """
    Configuration class for the track model.

    Attributes:
        model_config (dict): A dictionary containing the configuration options for the model.
        head_config (dict): A dictionary containing the configuration options for the head.
        neighbor_dim (int): The dimension of the neighbor.
        dtype (torch.dtype): The data type to use for the model.
        device (Union[str, torch.device]): The device to use for the model.
        kwargs (dict): Additional keyword arguments for the model.
        head_kwargs (dict): Additional keyword arguments for the head.
    """

    neighbor_dim: int
    head_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    dtype: torch.dtype = DTYPE
    device: Union[str, torch.device] = DEVICE
    kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Initializes the track model configuration.
        """
        if "dtype" not in self.kwargs:
            self.kwargs["dtype"] = self.dtype
        if "dtype" not in self.head_kwargs:
            self.head_kwargs["dtype"] = self.dtype

        if "device" not in self.kwargs:
            self.kwargs["device"] = self.device
        if "device" not in self.head_kwargs:
            self.head_kwargs["device"] = self.device

        self.model_config = ModelConfig(**self.model_config)
        self.head_config = ClassHeadConfig(**self.head_config)


@dataclass
class TrackingOutputs:
    """
    A class representing the outputs of the object tracking.

    Attributes:
    - outputs: The outputs of the object tracking model.
    - logits: The logits tensor produced by the object tracking model.
    """

    outputs: BaseModelOutputWithPooling
    logits: torch.Tensor


class TrackingModel(nn.Module):
    """
    A class representing a model for object tracking.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.

    Attributes:
        config (ModelConfig): An instance of the ModelConfig class containing the model configuration.
        model (AutoModel): The pretrained model used for object tracking.

    """

    def __init__(self, config: dict = {}):
        self.config = config = ModelConfig(**config)
        self.model = (
            AutoModel.from_pretrained(config.model_name)
            .to(config.dtype)
            .to(config.device)
        )
        self.head = ClassHead(config.head_config).to(config.dtype).to(config.device)

    def __call__(
        self, inputs: transformers.image_processing_utils.BatchFeature
    ) -> BaseModelOutputWithPooling:
        """
        Perform object tracking using the model.

        Args:
            inputs (BatchFeature): The input batch of features for object tracking.

        Returns:
            BaseModelOutputWithPooling: The output of the object tracking, including the pooled features.

        """
        outputs = self.model(
            **inputs.to(self.config.dtype).to(self.config.device), **self.config.kwargs
        )
        logits = self.head(outputs.last_hidden_state, **self.config.head_kwargs)
        return TrackingOutputs(outputs, logits)
