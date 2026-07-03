"""Model implementation for cluster classification."""

from torch import nn


class Model(nn.Module):
    """The structure of the classification model.

    Extends: `torch.nn.Module`
    """

    def __init__(self) -> None:
        """Create a new model."""
        super().__init__()
        # Other layers to try: Dropout and batch normalization,
        # if they make any sense. It's a small model though, so likely not
        self.stack = nn.Sequential(
            # Output based on what was provided by the CCD
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            # em similarity, tau similarity
        )

    def forward(self, x):  # noqa: ANN201 ANN001
        """Execute the forward pass.

        Args:
            x: The input vector for the model to process..

        Returns:
            _type_: The certainty of the model for each label.

        """
        certainties = self.stack(x)
        return certainties
