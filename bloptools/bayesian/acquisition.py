from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

import bluesky.plans as bp
import numpy as np


def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


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


