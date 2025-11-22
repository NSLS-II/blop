from typing import Literal, cast

from databroker import Broker
from tiled.client.container import Container

from .data_access import DataAccess, DatabrokerDataAccess, TiledDataAccess
from .objectives import Objective
from .protocols import EvaluationFunction


def default_evaluation_function(
    uid: str,
    suggestions: list[dict],
    *,
    data_access: DataAccess,
    objectives: list[Objective],
) -> list[dict]:
    """
    Simple evaluation function.

    Assumes the following:
    - Objective names are the same as the names of the columns in the dataframe.
    - Each row in the dataframe corresponds to a single trial.

    Parameters
    ----------
    uid: str
        The unique identifier of the Bluesky run to evaluate.
    suggestions: list[dict]
        A list of dictionaries, each containing the parameterization of a point.
        The "_id" key is optional and can be used to identify each suggestion.
    tiled_client: Container
        The tiled client to read the data from.
    active_objectives: list[Objective]
        The active objectives of the experiment.

    Returns
    -------
    list[dict]
        A dictionary mapping objective names to their mean and standard error. Since there
        is a single trial, the standard error is None.
        Each outcome dictionary contains an "_id" key to identify the suggestion that produced it.
    """
    data = data_access.get_data(uid)
    outcomes = []
    for suggestion in suggestions:
        id = cast(str, suggestion.get("_id", None))
        if id is None:
            raise ValueError(f"suggestion must have an '_id' key. Got: {suggestion}")
        outcome: dict[str, tuple[float, None] | int | Literal["baseline"]] = {}
        if id == "baseline":
            outcome.update({objective.name: (data[objective.name][0], None) for objective in objectives})
        elif isinstance(id, int):
            outcome.update(
                {objective.name: (data[objective.name][id % len(data[objective.name])], None) for objective in objectives}
            )
        else:
            raise ValueError(
                f"Invalid '_id' type for this evaluation function. Got: {type(id)} for suggestion: {suggestion}."
            )
        outcome["_id"] = id
        outcomes.append(outcome)

    return outcomes


class DataAccessEvaluationFunction(EvaluationFunction):
    def __init__(self, data_access: DataAccess, objectives: list[Objective]):
        self.data_access = data_access
        self.objectives: list[Objective] = objectives

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        return default_evaluation_function(uid, suggestions, data_access=self.data_access, objectives=self.objectives)


class TiledEvaluationFunction(DataAccessEvaluationFunction):
    def __init__(self, tiled_client: Container, objectives: list[Objective]):
        super().__init__(TiledDataAccess(tiled_client), objectives)


class DatabrokerEvaluationFunction(DataAccessEvaluationFunction):
    def __init__(self, databroker: Broker, objectives: list[Objective]):
        super().__init__(DatabrokerDataAccess(databroker), objectives)
