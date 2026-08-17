"""A simple, linear sequential ML model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ml_on_apx.model_management.group_info import Activation

if TYPE_CHECKING:
    from ml_on_apx.model_management.models.simple_model.simple_info import (
        SimpleGroupInfo,
    )


class SimpleModel(nn.Module):
    """A machine-learning model."""

    def __init__(self, group_info: SimpleGroupInfo) -> None:
        """Initialize a model."""
        super(SimpleModel, self).__init__()
        activations = Activation.get_activations()
        stack: list[nn.Module] = []
        start_size = group_info.get_layer_size(0)
        for i in range(1, group_info.layer_count):
            end_size = group_info.get_layer_size(i)
            stack.append(nn.Linear(start_size, end_size))
            stack.append(activations[group_info.get_layer_activation(i)].activation())
            start_size = end_size
        self.stack = nn.Sequential(*stack)
        self.mask = torch.Tensor(
            [  # TODO test masking system
                group_info.all_features[x] in group_info.features
                for x in range(len(group_info.all_features))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass.

        Args:
            x: The input vector for the model to process..

        Returns:
            Tensor: The certainty of the model for each label.

        """
        certainties = self.stack(x[self.mask])
        return certainties
