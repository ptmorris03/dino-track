from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from dinotrack.util import multi_getattr


@dataclass
class ClassHeadConfig:
    """
    Configuration class for the patch classification head.

    Attributes:
        num_classes (int): The number of classes.
        model_dim (int): The model dimension of the patch classification head.
        activation (Callable[[torch.Tensor], torch.Tensor]): The activation function to be used.
            Default is F.tanh.
        hidden_dim (list[int]): The hidden dimensions of the patch classification head.
            Default is an empty list.
    """

    num_classes: int
    model_dim: int
    bias: bool = True
    activation: Callable[[torch.Tensor], torch.Tensor] = F.tanh
    hidden_dim: list[int] = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the patch classification head configuration.
        """
        if not isinstance(self.hidden_dim, list):
            self.hidden_dim = (
                [self.hidden_dim]
                if isinstance(self.hidden_dim, int)
                else list(self.hidden_dim)
            )

        if isinstance(self.activation, str):
            self.activation = multi_getattr([F, nn, torch], self.activation)

        if isinstance(self.activation, type):
            self.activation = self.activation()

        assert callable(self.activation)

        # assert the number of non-self arguments to the activation() is 1
        cls_offset = 1  # assume an nn.Module class by default
        activation = getattr(self.activation, "forward", None)
        if activation is None:
            activation = self.activation
            cls_offset = 0  # function, not a class

        assert (activation.__code__.co_argcount - cls_offset) == 1

    @property
    def dims(self) -> list[int]:
        """
        Get the dimensions of the patch classification head.

        Returns:
            list[int]: The dimensions of the patch classification head.
        """
        return [self.model_dim] + self.hidden_dim + [self.num_classes]


class ClassHead(nn.Module):
    """
    A class representing a class head for object tracking.

    Args:
        config (dict): A dictionary containing the configuration parameters for the class head.

    Attributes:
        config (ClassHeadConfig): An instance of the ClassHeadConfig class containing the class head configuration.
    """

    def __init__(self, config: dict = {}):
        super(ClassHead, self).__init__()
        self.config = config = ClassHeadConfig(**config)
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out, bias=config.bias)
                for d_in, d_out in zip(config.dims[:-1], config.dims[1:])
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform object tracking using the class head.

        Args:
            inputs (torch.Tensor): The input tensor for object tracking.

        Returns:
            torch.Tensor: The output tensor of the patch classification
        """
        x = inputs
        for layer in self.layers[:-1]:
            x = self.config.activation(layer(x))
        return self.layers[-1](x)
