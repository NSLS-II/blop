import numpy as np

from blop.plans.utils import get_route_index, route_suggestions
from blop.protocols import ID_KEY


# get_route_index tests

def test_get_route_index_two_points_no_start():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = get_route_index(points)
    assert set(result) == {0, 1}


def test_get_route_index_multiple_points_no_start():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = get_route_index(points)
    assert set(result) == {0, 1, 2}


def test_get_route_index_with_starting_point():
    points = np.array([[1.0, 0.0], [2.0, 0.0]])
    start = np.array([0.0, 0.0])
    result = get_route_index(points, starting_point=start)
    assert set(result) == {0, 1}


# route_suggestions tests

def test_route_suggestions_single_returns_unchanged():
    suggestions = [{"x": 1.0, "y": 2.0, ID_KEY: "a"}]
    result = route_suggestions(suggestions)
    assert result == suggestions


def test_route_suggestions_multiple_no_start():
    suggestions = [
        {"x": 0.0, "y": 0.0, ID_KEY: "a"},
        {"x": 1.0, "y": 0.0, ID_KEY: "b"},
    ]
    result = route_suggestions(suggestions)
    assert len(result) == 2
    assert {s[ID_KEY] for s in result} == {"a", "b"}


def test_route_suggestions_multiple_with_start():
    suggestions = [
        {"x": 10.0, "y": 0.0, ID_KEY: "far"},
        {"x": 1.0, "y": 0.0, ID_KEY: "near"},
    ]
    start = {"x": 0.0, "y": 0.0}
    result = route_suggestions(suggestions, starting_position=start)
    # "near" should come first since it's closer to start
    assert result[0][ID_KEY] == "near"


def test_route_suggestions_ignores_non_float_values():
    suggestions = [
        {"x": 0.0, "label": "foo", ID_KEY: "a"},
        {"x": 1.0, "label": "bar", ID_KEY: "b"},
    ]
    result = route_suggestions(suggestions)
    assert len(result) == 2
