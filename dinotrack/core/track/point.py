from dataclasses import dataclass, field

import torch


@dataclass
class PointTracker:
    """
    A class representing a point tracker.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        label (int | str): The label of the point.
        alpha (float, optional): The alpha value for exponential moving average. Defaults to 0.5.
        device (str | torch.device | None, optional): The device to use for computation. Defaults to None.
        vector (torch.Tensor, optional): The vector representing the point. Defaults to None.
    """

    x: float
    y: float
    label: int | str
    alpha: float = 0.5
    device: str | torch.device | None = None
    vector: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the PointTracker object.

        If alpha is not a torch.Tensor, it is converted to a tensor with the specified device or "cpu".
        """
        if not isinstance(self.alpha, torch.Tensor):
            self.alpha = torch.tensor(self.alpha, device=self.device or "cpu")

    def update(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Updates the vector of the point tracker.

        Args:
            vector (torch.Tensor): The new vector to update.

        Returns:
            torch.Tensor: The updated vector.
        """
        # update the vector (exponential moving average)
        if hasattr(self, "vector"):
            alpha = self.alpha.to(self.vector.device)
            vector = vector.to(self.vector.device)
            variance_scale = torch.sqrt(self.alpha**2 + (1 - self.alpha) ** 2)
            self.vector = (alpha * vector + (1 - alpha) * self.vector) / variance_scale
            return self.vector

        # initialize the vector
        self.vector = vector.to(self.device or vector.device)
        return vector
