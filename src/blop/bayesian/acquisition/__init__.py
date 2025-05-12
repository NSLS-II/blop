import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from botorch.acquisition.acquisition import AcquisitionFunction  # type: ignore[import-untyped]

from . import analytic, monte_carlo
from .analytic import *  # noqa F401
from .monte_carlo import *  # noqa F401

if TYPE_CHECKING:
    from ...agent import BaseAgent

# from botorch.utils.transforms import normalize


here, this_filename = os.path.split(__file__)

# TODO: Move this into the code, rather than a config file
with open(f"{here}/config.yml") as f:
    config = yaml.safe_load(f)


def all_acqfs(columns: tuple[str, ...] = ("identifier", "type", "multitask_only", "description")) -> pd.DataFrame:
    cols = list(columns)
    acqfs = pd.DataFrame(config).T[cols]
    acqfs.index.name = "name"
    return acqfs.sort_values(["type", "name"])


def parse_acqf_identifier(identifier: str, strict: bool = True) -> dict[str, Any] | None:
    for acqf_name in config.keys():
        if identifier.lower() in [acqf_name, config[acqf_name]["identifier"]]:
            return {"name": acqf_name, **config[acqf_name]}
    if strict:
        raise ValueError(f"'{identifier}' is not a valid acquisition function identifier.")
    return None


def _construct_acqf(agent: "BaseAgent", acqf_name: str, **acqf_kwargs: Any) -> tuple[AcquisitionFunction, dict[str, Any]]:
    """Generates an acquisition function from a supplied identifier. A list of acquisition functions and
    their identifiers can be found at `agent.all_acqfs`.

    Args:
        agent: The optimization agent
        acqf_name: Name of the acquisition function
        **acqf_kwargs: Additional keyword arguments for the acquisition function

    Returns:
        tuple: (acquisition_function, acquisition_function_kwargs)
    """

    acqf_config = config["upper_confidence_bound"]

    if config[acqf_name]["multitask_only"] and (len(agent.objectives) == 1):
        raise ValueError(f'Acquisition function "{acqf_name}" is only for multi-task optimization problems!')

    # there is probably a better way to structure this
    if acqf_name == "expected_improvement":
        acqf_kwargs["best_f"] = agent.best_f(weights="default")

        acqf = analytic.ConstrainedLogExpectedImprovement(
            constraint=agent.constraint,
            model=agent.fitness_model,
            posterior_transform=agent.fitness_scalarization(weights="default"),
            **acqf_kwargs,
        )

    elif acqf_name == "monte_carlo_expected_improvement":
        acqf_kwargs["best_f"] = agent.best_f(weights="default")

        acqf = monte_carlo.qConstrainedExpectedImprovement(
            constraint=agent.constraint,
            model=agent.fitness_model,
            posterior_transform=agent.fitness_scalarization(weights="default"),
            **acqf_kwargs,
        )

    elif acqf_name == "probability_of_improvement":
        acqf_kwargs["best_f"] = agent.best_f(weights="default")

        acqf = analytic.ConstrainedLogProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.fitness_model,
            posterior_transform=agent.fitness_scalarization(),
            **acqf_kwargs,
        )

    elif acqf_name == "monte_carlo_probability_of_improvement":
        acqf = monte_carlo.qConstrainedProbabilityOfImprovement(
            constraint=agent.constraint,
            model=agent.fitness_model,
            best_f=agent.best_f(),
            posterior_transform=agent.fitness_scalarization(),
        )

    elif acqf_name == "lower_bound_max_value_entropy":
        acqf = monte_carlo.qConstrainedLowerBoundMaxValueEntropy(
            constraint=agent.constraint,
            model=agent.fitness_model,
            candidate_set=agent.sample(n=1024).squeeze(1),
        )

    elif acqf_name == "monte_carlo_noisy_expected_hypervolume_improvement":
        acqf_kwargs["ref_point"] = acqf_kwargs.get("ref_point", agent.random_ref_point)

        acqf = monte_carlo.qConstrainedNoisyExpectedHypervolumeImprovement(
            constraint=agent.constraint,
            model=agent.fitness_model,
            # X_baseline=agent.input_normalization.forward(agent.train_inputs())[],
            X_baseline=agent.dofs(active=True).transform(agent.train_inputs(active=True)),
            prune_baseline=True,
            **acqf_kwargs,
        )

    elif acqf_name == "upper_confidence_bound":
        acqf_kwargs["beta"] = acqf_kwargs.get("beta", acqf_config["default_kwargs"]["beta"])

        acqf = analytic.ConstrainedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.fitness_model,
            posterior_transform=agent.fitness_scalarization(),
            **acqf_kwargs,
        )

    elif acqf_name == "monte_carlo_upper_confidence_bound":
        acqf_kwargs["beta"] = acqf_kwargs.get("beta", acqf_config["default_kwargs"]["beta"])

        acqf = monte_carlo.qConstrainedUpperConfidenceBound(
            constraint=agent.constraint,
            model=agent.fitness_model,
            posterior_transform=agent.fitness_scalarization(),
            **acqf_kwargs,
        )

    elif acqf_name == "expected_mean":
        acqf, _ = _construct_acqf(agent, acqf_name="upper_confidence_bound", beta=0)
        acqf_kwargs = {}

    elif acqf_name == "monte_carlo_expected_mean":
        acqf, _ = _construct_acqf(agent, acqf_name="monte_carlo_upper_confidence_bound", beta=0)
        acqf_kwargs = {}

    return acqf, acqf_kwargs
