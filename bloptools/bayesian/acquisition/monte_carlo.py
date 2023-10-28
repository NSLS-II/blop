import math

import numpy as np
import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


class qConstrainedUpperConfidenceBound(qUpperConfidenceBound):
    def __init__(self, constraint, beta=4, *args, **kwargs):
        super().__init__(beta=beta, *args, **kwargs)
        self.constraint = constraint
        self.beta = torch.tensor(beta)

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


class qConstrainedExpectedImprovement(qExpectedImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedProbabilityOfImprovement(qProbabilityOfImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x).squeeze(-1)


class qConstrainedLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x).squeeze(-1)
