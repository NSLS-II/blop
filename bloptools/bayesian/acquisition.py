import bluesky.plans as bp
import numpy as np

import math
import torch

from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid



class ConstrainedUpperConfidenceBound(UpperConfidenceBound):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):

        mean, sigma = self._mean_and_sigma(x)

        p_eff = 0.5 * (1 + torch.special.erf(self.beta.sqrt()/math.sqrt(2))) * torch.clamp(self.constraint(x), min=1e-6)

        return (mean if self.maximize else -mean) + sigma * np.sqrt(2) * torch.special.erfinv(2*p_eff-1)



class ConstrainedLogExpectedImprovement(LogExpectedImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) + self.constraint(x).log()


class ConstrainedLogProbabilityOfImprovement(LogProbabilityOfImprovement):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        return super().forward(x) + self.constraint(x).log()


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


