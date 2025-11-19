from ax.api.types import TOutcome
from tiled.client.container import Container

from .data_access import TiledDataAccess
from .objectives import Objective


def default_evaluation_function(
    uid: str,
    trial_uids: set[int | str] | None = None,
    *,
    tiled_client: Container,
    active_objectives: list[Objective],
) -> TOutcome:
    """
    Simple evaluation function.

    Assumes the following:
    - Objective names are the same as the names of the columns in the dataframe.
    - Each row in the dataframe corresponds to a single trial.

    Parameters
    ----------
    uid: str
        The unique identifier of the Bluesky run to evaluate.
    trial_uids: set[int | str], optional
        The unique identifiers for the trials to evaluate. This is usually
        only one value, but can be a set of values if multiple suggestions
        are made at once.
    tiled_client: Container
        The tiled client to read the data from.
    active_objectives: list[Objective]
        The active objectives of the experiment.

    Returns
    -------
    TOutcome
        A dictionary mapping objective names to their mean and standard error. Since there
        is a single trial, the standard error is None.
    """
    if not trial_uids:
        raise ValueError(f"trial_uids must be provided for this evaluation function. Got: {trial_uids}")

    data = TiledDataAccess(tiled_client).get_data(uid)
    outcomes = []
    for trial_uid in trial_uids:
        outcome = {
            objective.name: (data[objective.name][trial_uid % len(data[objective.name])], None) for objective in active_objectives
        }
        outcome["_id"] = trial_uid
        outcomes.append(outcome)

    return outcomes
