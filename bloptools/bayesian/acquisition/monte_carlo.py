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
        posterior = self.model.posterior(x)
        mean, sigma = posterior.mean, posterior.variance.sqrt()

        p_eff = 0.5 * (1 + torch.special.erf(self.beta.sqrt() / math.sqrt(2))) * torch.clamp(self.constraint(x), min=1e-6)

        return mean + sigma * np.sqrt(2) * torch.special.erfinv(2 * p_eff - 1)


class qConstrainedExpectedImprovement(qExpectedImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstrainedProbabilityOfImprovement(qProbabilityOfImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstrainedNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstrainedLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)
