from typing import Any

import networkx as nx
import numpy as np

from ..protocols import ID_KEY


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


def retrieve_suggestions_from_user(actuators: list, optimizer) -> list[dict] | None:
    """
    Retrieve manual point suggestions from the user.

    Parameters
    ----------
    actuators : list
        The actuators to suggest values for.
    optimizer : Optimizer
        The optimizer containing parameter and objective configurations.

    Returns
    -------
    list[dict] | None
        A list of suggested points as dictionaries with DOF and objective values, or None if user declined.
    """
    # Get parameter configs and objectives from the optimizer
    objectives = optimizer.ax_client._experiment.optimization_config.objective.metrics
    model_components = actuators + objectives

    suggestions = []
    while True:
        # Build a single suggestion point with all DOF and objective values
        new_suggestion = {}
        for item in model_components:
            while True:
                try:
                    value = float(input(f"Enter value for {item.name} (float): "))
                    new_suggestion[item.name] = value
                    break
                except ValueError:
                    print(f"Invalid input. Please enter a valid number for {item.name}.")

        suggestions.append(new_suggestion)

        # Ask if user wants to add more points
        if (
            ask_user_for_input("Do you want to suggest another point?", options={"y": "Yes", "n": "No, finish suggestions"})
            == "n"
        ):
            break

    return suggestions
