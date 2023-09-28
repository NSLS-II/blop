from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


class qConstrainedExpectedImprovement(qExpectedImprovement):
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
