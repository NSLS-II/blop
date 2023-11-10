from typing import Union

import botorch
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover
from torch import Tensor


class TargetingPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors."""

    scalarize: bool = True

    def __init__(self, weights: Tensor, targets: Tensor) -> None:
        r"""
        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
            offset: An offset to be added to posterior mean.
        """
        super().__init__()
        self.targets = targets
        self.register_buffer("weights", weights)

    def sampled_transform(self, y):
        for i, target in enumerate(self.targets):
            if target == "min":
                y[..., i] = -y[..., i]
            elif target != "max":
                y[..., i] = -(y[..., i] - target).abs()

        return y @ self.weights.unsqueeze(-1)

    def mean_transform(self, mean, var):
        for i, target in enumerate(self.targets):
            if target == "min":
                mean[..., i] = -mean[..., i]
            elif target != "max":
                mean[..., i] = -(mean[..., i] - target).abs()

        return mean @ self.weights.unsqueeze(-1)

    def variance_transform(self, mean, var):
        return var @ self.weights.unsqueeze(-1)

    def evaluate(self, Y: Tensor) -> Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A `batch_shape x q x m`-dim tensor of outcomes.

        Returns:
            A `batch_shape x q`-dim tensor of transformed outcomes.
        """
        return self.sampled_transform(Y)

    def forward(self, posterior: Union[GPyTorchPosterior, PosteriorList]) -> GPyTorchPosterior:
        r"""Compute the posterior of the affine transformation.

        Args:
            posterior: A posterior with the same number of outputs as the
                elements in `self.weights`.

        Returns:
            A single-output posterior.
        """

        return botorch.posteriors.transformed.TransformedPosterior(
            posterior,
            sample_transform=self.sampled_transform,
            mean_transform=self.mean_transform,
            variance_transform=self.variance_transform,
        )
