import time

from bluesky.protocols import Readable, Reading, SyncOrAsync
from event_model import DataKey
import networkx as nx
import numpy as np

from ..protocols import ID_KEY


class NumberReadable(Readable[float]):
    def __init__(self, name: str, value: float = 0.0):
        self.name: str = name
        self._value = value

    def read(self) -> SyncOrAsync[dict[str, Reading[float]]]:
        return {self.name: {"value": self._value, "timestamp": time.time()}}

    def describe(self) -> SyncOrAsync[dict[str, DataKey]]:
        return {self.name: {"source": self.name, "dtype": "number", "shape": []}}


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
