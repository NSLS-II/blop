import networkx as nx
import numpy as np


def get_route_index(points: np.ndarray, cycle: bool = False):
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    for i, i_point in enumerate(points):
        for j, j_point in enumerate(points):
            if i >= j:
                continue
            G.add_weighted_edges_from([(i, j, np.sqrt(np.sum(np.square(i_point - j_point))))])

    return np.array(nx.approximation.traveling_salesman_problem(G, cycle=cycle))


def route_suggestions(suggestions: list[dict], cycle: bool = False):
    dims = [dim for dim in suggestions[0] if dim != "_id"]
    points = np.array([[s[dim] for dim in dims] for s in suggestions])

    return [suggestions[i] for i in get_route_index(points=points, cycle=cycle)]
