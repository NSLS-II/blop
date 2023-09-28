import math

import numpy as np
import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


class ConstraintedUpperConfidenceBound(qUpperConfidenceBound):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        mean, sigma = self._mean_and_sigma(x)

        p_eff = 0.5 * (1 + torch.special.erf(self.beta.sqrt() / math.sqrt(2))) * torch.clamp(self.constraint(x), min=1e-6)

        return (mean if self.maximize else -mean) + sigma * np.sqrt(2) * torch.special.erfinv(2 * p_eff - 1)


class qConstraintedExpectedImprovement(qExpectedImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstraintedProbabilityOfImprovement(qProbabilityOfImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstraintedNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)


class qConstraintedLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) * self.constraint(x)
