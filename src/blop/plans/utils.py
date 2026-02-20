import time
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from bluesky.protocols import HasHints, HasParent, Hints, Readable, Reading
from event_model import DataKey
from numpy.typing import ArrayLike

from ..protocols import ID_KEY, OptimizationProblem


def _infer_data_key(value: ArrayLike) -> DataKey:
    """Infer the data key from the provided value."""
    numpy_array = np.array(value)
    dtype_numpy = numpy_array.dtype.str
    if len(numpy_array.shape) > 1 or (len(numpy_array.shape) == 1 and numpy_array.shape[0] > 1):
        dtype = "array"
        shape = list(numpy_array.shape)
    else:
        shape = []
        item = numpy_array[0] if len(numpy_array.shape) == 1 else numpy_array.item()
        if isinstance(item, (int, float)):
            dtype = "number"
        else:
            dtype = "string"
    return DataKey(source="blop_optimization", dtype=dtype, shape=shape, dtype_numpy=dtype_numpy)


class InferredReadable(Readable, HasHints, HasParent):
    """
    An inferred readable object that can be used in Bluesky plans.

    It performs inference on the initial value to describe the data key.

    Parameters
    ----------
    name : str
        The name of the readable instance.
    initial_value : numpy.typing.ArrayLike
        The initial value of the readable instance.
    """

    def __init__(self, name: str, initial_value: ArrayLike) -> None:
        self._name = name
        self._data_key = None

        if isinstance(initial_value, np.ndarray):
            self._dtype = initial_value.dtype
            initial_value = initial_value.tolist()
        else:
            self._dtype = None

        if isinstance(initial_value, Sequence) and len(initial_value) == 1:
            initial_value = initial_value[0]
        self._value = initial_value

    @property
    def parent(self) -> Any | None:
        return None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return {
            "fields": [self.name],
            "dimensions": [],
            "gridding": "rectilinear",
        }

    def describe(self) -> dict[str, DataKey]:
        if not self._data_key:
            # Use stored dtype if available, otherwise infer
            if self._dtype is not None:
                numpy_array = np.array(self._value, dtype=self._dtype)
            else:
                numpy_array = np.array(self._value)
            self._data_key = _infer_data_key(numpy_array)
        return {self.name: self._data_key}

    def update(self, value: ArrayLike) -> None:
        if isinstance(value, np.ndarray):
            self._dtype = value.dtype
            value = value.tolist()
        else:
            self._dtype = None

        if isinstance(value, Sequence) and len(value) == 1:
            value = value[0]
        self._value = value

    def read(self) -> dict[str, Reading]:
        return {
            self.name: {
                "value": self._value,
                "timestamp": time.time(),
            }
        }


def get_route_index(points: np.ndarray, starting_point: np.ndarray | None = None):
    if starting_point is not None:
        points = np.concatenate([starting_point[None], points], axis=0)

    G = nx.DiGraph()
    for i, i_point in enumerate(points):
        for j, j_point in enumerate(points):
            if j <= i:
                continue
            d = np.sqrt(np.sum(np.square(i_point - j_point)))
            G.add_edge(i, j, weight=d)
            G.add_edge(j, i, weight=d if i > 0 else 1e2 * d)

    index = nx.approximation.traveling_salesman_problem(
        G, cycle=False, method=nx.approximation.simulated_annealing_tsp, init_cycle="greedy"
    )

    if starting_point is not None:
        index = [i - 1 for i in index if i > 0]
    return index


def route_suggestions(suggestions: list[dict], starting_position: dict | None = None):
    if len(suggestions) == 1:
        return suggestions

    dims_to_route = [dim for dim, value in suggestions[0].items() if (dim != ID_KEY) and isinstance(value, float)]
    points = np.array([[s[dim] for dim in dims_to_route] for s in suggestions])
    starting_point = np.array([starting_position[dim] for dim in dims_to_route]) if starting_position else None

    return [suggestions[i] for i in get_route_index(points=points, starting_point=starting_point)]


def ask_user_for_input(prompt: str, options: dict | None = None) -> Any:
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print()
    print(f"{BOLD}{BLUE}")
    print("+" + "-" * max((len(prompt) + 2 * 2), 58) + "+")
    print(f"| {prompt}".ljust(max((len(prompt) + 2 * 2), 58)) + " |")
    print("+" + "-" * max((len(prompt) + 2 * 2), 58) + "+")
    print(RESET, end="")
    if options is not None:
        for key, value in options.items():
            print(f"  {BOLD}{key}{RESET}: {value}")

        while True:
            # Build the prompt string with keys dynamically
            valid_keys = list(options.keys())
            keys_prompt = ",".join(valid_keys)
            choice = input(f"\nEnter choice [{keys_prompt}]: ").lower().strip()
            if choice in options:
                user_input = choice
                break
            print("Invalid selection.")
    else:
        user_input = input("> ")
    return user_input


def retrieve_suggestions_from_user(
    optimization_problem: OptimizationProblem,
    *args: Any,
    **kwargs: Any,
):
    """
    Retrieve manual point suggestions from the user.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    """
    from .plans import default_acquire

    dictonary_string = input(
        "Enter list of suggestions as a list of dictionaries (e.g., [{'x1': 1.0, 'x2': 2.0}, {'x1': 3.0, 'x2': 4.0}]): "
    )
    suggestions = ast.literal_eval(dictonary_string)

    optimizer = optimization_problem.optimizer

    # Manually attach trials
    for suggestion in suggestions:
        trial_idx = optimizer._client.attach_trial(parameters=suggestion)  # type: ignore[attr-defined]
        suggestion[ID_KEY] = trial_idx

    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan

    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    uid = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    outcomes = optimization_problem.evaluation_function(uid, suggestions)
    optimizer.ingest(outcomes)
    return suggestions
