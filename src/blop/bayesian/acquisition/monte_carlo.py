import math
from collections.abc import Callable

import numpy as np
import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy  # type: ignore[import-untyped]
from botorch.acquisition.monte_carlo import (  # type: ignore[import-untyped]
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.monte_carlo import (  # type: ignore[import-untyped]
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model import Model  # type: ignore[import-untyped]
from torch import Tensor


class qConstrainedUpperConfidenceBound(qUpperConfidenceBound):
    """Monte Carlo expected improvement, but scaled by some constraint.
    NOTE: Because the UCB can be negative, we constrain it by adjusting the Gaussian quantile.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, constraint: Callable[[Tensor], Tensor], beta: float = 4, **kwargs) -> None:
        super().__init__(beta=beta, **kwargs)
        self.constraint = constraint
        self.beta = torch.tensor(beta)

    def forward(self, x: Tensor) -> Tensor:
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


class qConstrainedExpectedImprovement(qExpectedImprovement):
    """Monte Carlo expected improvement, but scaled by some constraint.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model: Model, constraint: Callable[[Tensor], Tensor], **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedProbabilityOfImprovement(qProbabilityOfImprovement):
    """Monte Carlo probability of improvement, but scaled by some constraint.

    Parameters
    ----------
    model:
        A BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model: Model, constraint: Callable[[Tensor], Tensor], **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """Monte Carlo noisy expected hypervolume improvement, but scaled by some constraint.
    Only works with multi-objective models.

    Parameters
    ----------
    model:
        A multi-objective BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model: Model, constraint: Callable[[Tensor], Tensor], **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    """GIBBON (General-purpose Information-Based Bayesian OptimisatioN), but scaled by some constraint.

    Parameters
    ----------
    model:
        A multi-objective BoTorch model over which to compute the acquisition function.
    constraint:
        A callable which when evaluated on inputs returns the probability of feasibility.
    """

    def __init__(self, model: Model, constraint: Callable[[Tensor], Tensor], **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.constraint = constraint

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.constraint(x).squeeze(-1)
