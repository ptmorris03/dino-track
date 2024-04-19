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


@dataclass
class TrackModelConfig:
    """
    Configuration class for the track model.

    Attributes:
        model_config (dict): A dictionary containing the configuration options for the model.
    """

    neighbor_dim: int
    head_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Initializes the track model configuration.
        """
        self.model_config = ModelConfig(**self.model_config)
        self.head_config = ClassHeadConfig(**self.head_config)
