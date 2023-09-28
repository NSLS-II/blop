from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


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
