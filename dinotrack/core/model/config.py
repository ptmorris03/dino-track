from dataclasses import dataclass, field

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
