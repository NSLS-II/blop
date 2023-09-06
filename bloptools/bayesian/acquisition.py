import math
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import numpy as np
import torch
import time
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


# def default_acquisition_plan(dofs, inputs, dets):

#     args = []
#     for dof, points in zip(dofs, np.atleast_2d(inputs).T):
#         args.append(dof)
#         args.append(list(points))

#     uid = yield from bp.list_scan(dets, *args)
#     return uid


def list_scan_with_delay(*args, delay=0, **kwargs):
    "Accepts all the normal 'scan' parameters, plus an optional delay."

    def one_nd_step_with_delay(detectors, step, pos_cache):
        "This is a copy of bluesky.plan_stubs.one_nd_step with a sleep added."
        motors = step.keys()
        yield from bps.move_per_step(step, pos_cache)
        yield from bps.sleep(delay)
        yield from bps.trigger_and_read(list(detectors) + list(motors))

    kwargs.setdefault('per_step', one_nd_step_with_delay)
    uid = yield from bp.list_scan(*args, **kwargs)
    return uid


def default_acquisition_plan(dofs, inputs, dets):

    args = []
    for dof, points in zip(dofs, np.atleast_2d(inputs).T):
        args.append(dof)
        args.append(list(points))

    uid = yield from list_scan_with_delay(dets, *args, delay=1)
    return uid


# def sleepy_acquisition_plan(dofs, inputs, dets):

#     args = []
#     for dof, points in zip(dofs, np.atleast_2d(inputs).T):
#         args.append(dof)
#         args.append(list(points))

#     for point in inputs:
#         args = []
#         for dof, value in zip(dofs, point):
#             args.append(dof)
#             args.append(value)

#         yield from bps.mv(*args)
#         yield from bps.count([*dets, *dofs])
#         yield from bps.sleep(1)
    
#     return uid


ACQ_FUNC_CONFIG = {
    "quasi-random": {
        "identifiers": ["qr", "quasi-random"],
        "pretty_name": "Quasi-random",
        "description": "Sobol-sampled quasi-random points.",
        "multitask_only": False,
    },
    "expected_mean": {
        "identifiers": ["em", "expected_mean"],
        "pretty_name": "Expected mean",
        "multitask_only": False,
        "description": "The expected value at each input.",
    },
    "expected_improvement": {
        "identifiers": ["ei", "expected_improvement"],
        "pretty_name": "Expected improvement",
        "multitask_only": False,
        "description": r"The expected value of max(f(x) - \nu, 0), where \nu is the current maximum.",
    },
    "noisy_expected_hypervolume_improvement": {
        "identifiers": ["nehvi", "noisy_expected_hypervolume_improvement"],
        "pretty_name": "Noisy expected hypervolume improvement",
        "multitask_only": True,
        "description": r"It's like a big box. How big is the box?",
    },
    "lower_bound_max_value_entropy": {
        "identifiers": ["lbmve", "lbmes", "gibbon", "lower_bound_max_value_entropy"],
        "pretty_name": "Lower bound max value entropy",
        "multitask_only": False,
        "description": r"Max entropy search, basically",
    },
    "probability_of_improvement": {
        "identifiers": ["pi", "probability_of_improvement"],
        "pretty_name": "Probability of improvement",
        "multitask_only": False,
        "description": "The probability that this input improves on the current maximum.",
    },
    "upper_confidence_bound": {
        "identifiers": ["ucb", "upper_confidence_bound"],
        "default_args": {"beta": 4},
        "pretty_name": "Upper confidence bound",
        "multitask_only": False,
        "description": r"The expected value, plus some multiple of the uncertainty (typically \mu + 2\sigma).",
    },
}


class ConstrainedUpperConfidenceBound(UpperConfidenceBound):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint

    def forward(self, x):
        mean, sigma = self._mean_and_sigma(x)

        p_eff = 0.5 * (1 + torch.special.erf(self.beta.sqrt() / math.sqrt(2))) * torch.clamp(self.constraint(x), min=1e-6)

        return (mean if self.maximize else -mean) + sigma * np.sqrt(2) * torch.special.erfinv(2 * p_eff - 1)


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
