from ax.api.types import TOutcome
from tiled.client.container import Container

from .data_access import TiledDataAccess
from .objectives import Objective


def default_evaluation_function(
    uid: str,
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
    data = TiledDataAccess(tiled_client).get_data(uid)
    return [
        {objective.name: (data[objective.name][trial_index], None) for objective in active_objectives}
        for trial_index in range(len(data[active_objectives[0].name]))
    ]
