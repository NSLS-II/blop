import os

import yaml
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from . import analytic, monte_carlo
from .analytic import *  # noqa F401
from .monte_carlo import *  # noqa F401

here, this_filename = os.path.split(__file__)

with open(f"{here}/config.yml", "r") as f:
    config = yaml.safe_load(f)


# supplying the full name is also a valid identifier
for acq_func_name in config.keys():
    config[acq_func_name]["identifiers"].append(acq_func_name)


def parse_acq_func(acq_func_identifier):
    acq_func_name = None
    for _acq_func_name in config.keys():
        if acq_func_identifier.lower() in config[_acq_func_name]["identifiers"]:
            acq_func_name = _acq_func_name

    if acq_func_name is None:
        raise ValueError(f'Unrecognized acquisition function identifier "{acq_func_identifier}".')

    return acq_func_name


def get_acquisition_function(agent, acq_func_identifier="qei", return_metadata=False, **acq_func_kwargs):
    """
    Generates an acquisition function from a supplied identifier.
    """

    acq_func_name = parse_acq_func(acq_func_identifier)
    acq_func_config = agent.acq_func_config["upper_confidence_bound"]

    if agent.acq_func_config[acq_func_name]["multitask_only"] and (agent.num_tasks == 1):
        raise ValueError(f'Acquisition function "{acq_func_name}" is only for multi-task optimization problems!')

    # there is probably a better way to structure this
    if acq_func_name == "expected_improvement":
        acq_func = analytic.ConstraintedLogExpectedImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.best_scalarized_fitness,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "monte_carlo_expected_improvement":
        acq_func = monte_carlo.qConstraintedExpectedImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.best_scalarized_fitness,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "probability_of_improvement":
        acq_func = analytic.ConstraintedLogProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.best_scalarized_fitness,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "monte_carlo_probability_of_improvement":
        acq_func = monte_carlo.qConstraintedProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.best_scalarized_fitness,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "lower_bound_max_value_entropy":
        acq_func = monte_carlo.qConstraintedLowerBoundMaxValueEntropy(
            constraint=agent.constraint,
            model=agent.model,
            candidate_set=agent.test_inputs(n=1024).squeeze(1),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "noisy_expected_hypervolume_improvement":
        acq_func = monte_carlo.qConstraintedNoisyExpectedHypervolumeImprovement(
            constraint=agent.constraint,
            model=agent.model,
            ref_point=agent.train_targets.min(dim=0).values,
            X_baseline=agent.train_inputs,
            prune_baseline=True,
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "upper_confidence_bound":
        beta = acq_func_kwargs.get("beta", acq_func_config["default_args"]["beta"])

        acq_func = analytic.ConstraintedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.model,
            beta=beta,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {"beta": beta}}

    elif acq_func_name == "monte_carlo_upper_confidence_bound":
        beta = acq_func_kwargs.get("beta", acq_func_config["default_args"]["beta"])

        acq_func = monte_carlo.qConstraintedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.model,
            beta=beta,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.task_weights, offset=0),
            **acq_func_kwargs,
        )
        acq_func_meta = {"name": acq_func_name, "args": {"beta": beta}}

    elif acq_func_name == "expected_mean":
        acq_func = agent.get_acquisition_function(acq_func_identifier="ucb", beta=0, return_metadata=False)
        acq_func_meta = {"name": acq_func_name, "args": {}}

    return (acq_func, acq_func_meta) if return_metadata else acq_func
