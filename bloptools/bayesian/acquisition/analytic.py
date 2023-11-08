import math

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement, UpperConfidenceBound


def list_scan_with_delay(*args, delay=0, **kwargs):
    "Accepts all the normal 'scan' parameters, plus an optional delay."

    def one_nd_step_with_delay(detectors, step, pos_cache):
        "This is a copy of bluesky.plan_stubs.one_nd_step with a sleep added."
        motors = step.keys()
        yield from bps.move_per_step(step, pos_cache)
        yield from bps.sleep(delay)
        yield from bps.trigger_and_read(list(detectors) + list(motors))

    kwargs.setdefault("per_step", one_nd_step_with_delay)
    uid = yield from bp.list_scan(*args, **kwargs)
    return uid


def default_acquisition_plan(dofs, inputs, dets, **kwargs):
    delay = kwargs.get("delay", 0)
    args = []
    for dof, points in zip(dofs, np.atleast_2d(inputs).T):
        args.append(dof)
        args.append(list(points))

    uid = yield from list_scan_with_delay(dets, *args, delay=delay)
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
