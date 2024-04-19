from dataclasses import dataclass, field

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
