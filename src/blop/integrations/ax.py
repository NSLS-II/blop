import copy
from collections.abc import Callable
from typing import Any

import pandas as pd
from ax.service.ax_client import AxClient
from bluesky.plans import list_scan
from bluesky.protocols import NamedMovable, Readable


def create_blop_experiment(ax_client: AxClient, parameters: list[dict[str, Any]], *args, **kwargs) -> None:
    # Check that a movable key is present
    if not all("movable" in p for p in parameters):
        raise ValueError("All parameters must have a 'movable' key.")

    # Check that a name attribute is present
    if not all(hasattr(p["movable"], "name") for p in parameters):
        raise ValueError("All 'movable' values must have a 'name' attribute.")

    ax_parameters = copy.copy(parameters)
    for p in ax_parameters:
        p["name"] = p["movable"].name
        del p["movable"]

    ax_client.create_experiment(*args, parameters=ax_parameters, **kwargs)


def create_bluesky_evaluator(
    RE,
    db,
    readables: list[Readable],
    movables: list[NamedMovable],
    evaluation_function: Callable[[pd.DataFrame], dict[str, tuple[float, float]]],
    plan: Callable | None = None,
) -> Callable:
    """
    Create an evaluation function that runs a Bluesky plan and evaluates objectives.

    Parameters:
    -----------
    RE : RunEngine
        The Bluesky RunEngine
    db : databroker
        The databroker/tiled instance
    movables : List
        List of Bluesky motors/devices to optimize
    detectors : List
        List of Bluesky detectors to read
    evaluation_function : Callable[[pd.DataFrame], Dict[str, Tuple[float, float]]]
        Function that takes a dataframe from databroker and returns
        a dictionary mapping objective names to (mean, sem) tuples
    plan : Callable, optional
        Custom Bluesky plan to use. If None, uses list_scan

    Returns:
    --------
    Callable
        Function that takes an Ax parameterization and returns objective values
    """
    plan_function = plan or list_scan

    def evaluate(parameterization: dict[str, float] | dict[str, list[float]]) -> dict[str, tuple[float, float]]:
        # Prepare the parameters for the plan
        unpacked = []
        for m in movables:
            if m.name in parameterization:
                unpacked.append(m)
                if isinstance(parameterization[m.name], float):
                    unpacked.append([parameterization[m.name]])
                elif isinstance(parameterization[m.name], list):
                    unpacked.append(parameterization[m.name])
                else:
                    raise ValueError(f"Parameter {m.name} must be a float or list of floats.")
            else:
                raise ValueError(f"Parameter {m.name} not found in parameterization. Parameterization: {parameterization}")

        # Run the plan
        uid = RE(plan_function(readables, *unpacked))

        # Fetch the data
        results_df = db[uid][0].table(fill=True)

        # Evaluate the data
        return evaluation_function(results_df)

    return evaluate
