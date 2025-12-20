import networkx as nx
import numpy as np


def route_suggestions(suggestions):
    G = nx.Graph()
    G.add_nodes_from(range(len(suggestions)))

    for i, i_sug in enumerate(suggestions):
        for j, j_sug in enumerate(suggestions):
            if i >= j:
                continue
            d2 = 0
            for dim in i_sug:
                if dim != "_id":
                    d2 += (i_sug[dim] - j_sug[dim]) ** 2
            G.add_weighted_edges_from([(i, j, np.sqrt(d2))])

    return [suggestions[i] for i in nx.approximation.traveling_salesman_problem(G, cycle=False)]
