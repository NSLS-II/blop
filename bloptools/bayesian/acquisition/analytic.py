import math

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement, UpperConfidenceBound


class ConstrainedUpperConfidenceBound(UpperConfidenceBound):
    """Upper confidence bound, but scaled by some constraint.
    NOTE: Because the UCB can be negative, we constrain it by adjusting the Gaussian quantile.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model, constraint, **kwargs):
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        *input_shape, _, _ = x.shape
        transformed_posterior = self.posterior_transform(self.model.posterior(x))
        mean = transformed_posterior.mean.reshape(input_shape)
        sigma = transformed_posterior.variance.sqrt().reshape(input_shape)

        p_eff = (
            0.5
            * (1 + torch.special.erf(self.beta.sqrt() / math.sqrt(2)))
            * torch.clamp(self.constraint(x).reshape(input_shape), min=1e-6)
        )

        return mean + sigma * np.sqrt(2) * torch.special.erfinv(2 * p_eff - 1)


class ConstrainedLogExpectedImprovement(LogExpectedImprovement):
    """Log expected improvement, but scaled by some constraint.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model, constraint, **kwargs):
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return (super().forward(x) + self.constraint(x).log().squeeze(-1)).exp()


class ConstrainedLogProbabilityOfImprovement(LogProbabilityOfImprovement):
    """Log probability of improvement acquisition function, but scaled by some constraint.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model, constraint, **kwargs):
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return (super().forward(x) + self.constraint(x).log().squeeze(-1)).exp()
