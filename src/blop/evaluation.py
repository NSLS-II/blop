from ax.api.types import TOutcome
from tiled.client.container import Container

from .data_access import TiledDataAccess
from .objectives import Objective


def default_evaluation_function(
    trial_index: int,
    uid: str,
    tiled_client: Container,
    *,
    active_objectives: list[Objective],
) -> TOutcome:
    """
    Simple evaluation function.

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
    data = TiledDataAccess(tiled_client).get_data(uid)
    return {
        objective.name: (data[objective.name][(trial_index % len(data[objective.name]))], None)
        for objective in active_objectives
    }
