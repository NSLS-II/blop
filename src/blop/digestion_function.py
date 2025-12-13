import warnings
from typing import Any

from ax.api.types import TOutcome

from .objectives import Objective

warnings.warn(
    "The digestion_function module is deprecated and will be removed in Blop v1.0.0. "
    "See `blop.protocols.EvaluationFunction` as an alternative.",
    DeprecationWarning,
    stacklevel=2,
)


def default_digestion_function(
    trial_index: int,
    data: dict[str, list[Any]],
    *,
    active_objectives: list[Objective],
) -> TOutcome:
    """
    Simple digestion function.

    Assumes the following:
    - Objective names are the same as the names of the columns in the dataframe.
    - Each row in the dataframe corresponds to a single trial.

    Parameters
    ----------
    trial_index : int
        The index of the trial.
    data : dict[str, list[Any]]
        A dictonary containing the results of the experiment.
    active_objectives : list[Objective]
        The active objectives of the experiment.

    Returns
    -------
    TOutcome
        A dictionary mapping objective names to their mean and standard error. Since there
        is a single trial, the standard error is None.
    """
    return {
        objective.name: (data[objective.name][(trial_index % len(data[objective.name]))], None)
        for objective in active_objectives
    }
