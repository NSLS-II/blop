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


def parse_acq_func_identifier(identifier):
    acq_func_name = None
    for _acq_func_name in config.keys():
        if identifier.lower() in config[_acq_func_name]["identifiers"]:
            acq_func_name = _acq_func_name

    if acq_func_name is None:
        raise ValueError(f'Unrecognized acquisition function identifier "{identifier}".')

    return acq_func_name


def get_acquisition_function(agent, identifier="qei", return_metadata=True, verbose=False, **acq_func_kwargs):
    """Generates an acquisition function from a supplied identifier. A list of acquisition functions and
    their identifiers can be found at `agent.all_acq_funcs`.
    """

    acq_func_name = parse_acq_func_identifier(identifier)
    acq_func_config = config["upper_confidence_bound"]

    if config[acq_func_name]["multitask_only"] and (agent.num_tasks == 1):
        raise ValueError(f'Acquisition function "{acq_func_name}" is only for multi-task optimization problems!')

    # there is probably a better way to structure this
    if acq_func_name == "expected_improvement":
        acq_func = analytic.ConstrainedLogExpectedImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.max_scalarized_objective,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "monte_carlo_expected_improvement":
        acq_func = monte_carlo.qConstrainedExpectedImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.max_scalarized_objective,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "probability_of_improvement":
        acq_func = analytic.ConstrainedLogProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.max_scalarized_objective,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "monte_carlo_probability_of_improvement":
        acq_func = monte_carlo.qConstrainedProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.model,
            best_f=agent.max_scalarized_objective,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "lower_bound_max_value_entropy":
        acq_func = monte_carlo.qConstrainedLowerBoundMaxValueEntropy(
            constraint=agent.constraint,
            model=agent.model,
            candidate_set=agent.test_inputs(n=1024).squeeze(1),
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "noisy_expected_hypervolume_improvement":
        acq_func = monte_carlo.qConstrainedNoisyExpectedHypervolumeImprovement(
            constraint=agent.constraint,
            model=agent.model,
            ref_point=agent.train_targets.min(dim=0).values,
            X_baseline=agent.train_inputs,
            prune_baseline=True,
        )
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "upper_confidence_bound":
        beta = acq_func_kwargs.get("beta", acq_func_config["default_args"]["beta"])

        acq_func = analytic.ConstrainedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.model,
            beta=beta,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {"beta": beta}}

    elif acq_func_name == "monte_carlo_upper_confidence_bound":
        beta = acq_func_kwargs.get("beta", acq_func_config["default_args"]["beta"])

        acq_func = monte_carlo.qConstrainedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.model,
            beta=beta,
            posterior_transform=ScalarizedPosteriorTransform(weights=agent.objective_weights_torch, offset=0),
        )
        acq_func_meta = {"name": acq_func_name, "args": {"beta": beta}}

    elif acq_func_name == "expected_mean":
        acq_func = get_acquisition_function(agent, identifier="ucb", beta=0, return_metadata=False)
        acq_func_meta = {"name": acq_func_name, "args": {}}

    elif acq_func_name == "monte_carlo_expected_mean":
        acq_func = get_acquisition_function(agent, identifier="qucb", beta=0, return_metadata=False)
        acq_func_meta = {"name": acq_func_name, "args": {}}

    return (acq_func, acq_func_meta) if return_metadata else acq_func
