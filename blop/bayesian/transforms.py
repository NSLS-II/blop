from typing import Union

import botorch
import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList
from torch.special import erf

sqrt2 = np.sqrt(2)


def targeting_sample_transform(y: torch.Tensor, target) -> torch.Tensor:
    if target == "max":
        return y
    if target == "min":
        return -y
    elif not isinstance(target, tuple):
        return -(y - target).abs()
    else:
        return y * 0  # torch.where((y > target[0]) & (y < target[1]), 0, -np.inf)


def targeting_mean_transform(mean: torch.Tensor, variance: torch.Tensor, target) -> torch.Tensor:
    if target == "max":
        return mean
    if target == "min":
        return -mean
    elif not isinstance(target, tuple):
        return -(mean - target).abs()
    else:
        s = variance.sqrt()
        return torch.log(0.5 * (erf((target[1] - mean) / (sqrt2 * s)) - erf((target[0] - mean) / (sqrt2 * s))))
    # else:
    #    return -((mean - 0.5 * (target[1] + target[0])).abs() - 0.5 * (target[1] - target[0])).clamp(min=0)


def targeting_variance_transform(mean: torch.Tensor, variance: torch.Tensor, target) -> torch.Tensor:
    if isinstance(target, tuple):
        return 0
    else:
        return variance


class TargetingPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors."""

    scalarize: bool = True

    def __init__(self, weights: torch.Tensor, targets: torch.Tensor) -> None:
        r"""
        Args:
            weights: A one-dimensional torch.Tensor with `m` elements representing the
                linear weights on the outputs.
            offset: An offset to be added to posterior mean.
        """
        super().__init__()
        self.targets = targets
        self.register_buffer("weights", weights)

    def sample_transform(self, y):
        for i, target in enumerate(self.targets):
            y[..., i] = targeting_sample_transform(y[..., i], target)
        return y @ self.weights.unsqueeze(-1)

    def mean_transform(self, mean, var):
        for i, target in enumerate(self.targets):
            mean[..., i] = targeting_mean_transform(mean[..., i], var[..., i], target)
        return mean @ self.weights.unsqueeze(-1)

    def variance_transform(self, mean, var):
        for i, target in enumerate(self.targets):
            mean[..., i] = targeting_variance_transform(mean[..., i], var[..., i], target)
        return mean @ self.weights.unsqueeze(-1)

    def evaluate(self, Y: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A `batch_shape x q x m`-dim torch.Tensor of outcomes.

        Returns:
            A `batch_shape x q`-dim torch.Tensor of transformed outcomes.
        """
        return self.sample_transform(Y)

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
            sample_transform=self.sample_transform,
            mean_transform=self.mean_transform,
            variance_transform=self.variance_transform,
        )
