"""Model implementation for cluster identificaiton."""

from torch import nn


class Model(nn.Module):
    """The structure of the classification model."""

    def __init__(self) -> None:
        """Create a new model."""
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Define the network once data has been acquired
        )

    def forward(self, x):  # noqa: ANN201 ANN001
        """Execute the forward pass.

        Args:
            x: The input vector for the model to process..

        Returns:
            _type_: The certainty of the model for each label.

        """
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
