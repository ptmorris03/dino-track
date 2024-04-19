from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass
class ClassHeadConfig:
    """
    Configuration class for the class head.

    Attributes:
        num_classes (int): The number of classes.
        hidden_dim (int): The hidden dimension of the class head.
    """

    num_classes: int
    hidden_dim: list[int] = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the class head configuration.
        """
        if not isinstance(self.hidden_dim, list):
            self.hidden_dim = [self.hidden_dim]


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform object tracking using the class head.

        Args:
            inputs (torch.Tensor): The input tensor for object tracking.

        Returns:
            torch.Tensor: The output tensor of the object tracking.
        """
        # perform object tracking
        return inputs
