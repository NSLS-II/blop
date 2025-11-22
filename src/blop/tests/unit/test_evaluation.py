from unittest.mock import MagicMock, patch

import pytest
from databroker import Broker
from tiled.client.container import Container

from blop.data_access import DataAccess, DatabrokerDataAccess, TiledDataAccess
from blop.evaluation import (
    DataAccessEvaluationFunction,
    DatabrokerEvaluationFunction,
    TiledEvaluationFunction,
    default_evaluation_function,
)
from blop.objectives import Objective


@pytest.fixture(scope="function")
def mock_data_access():
    return MagicMock(spec=DataAccess)


def test_default_evaluation_function(mock_data_access):
    suggestions = [
        {
            "_id": 1,
            "x1": 0.0,
            "x2": 0.0,
        },
    ]
    objective = Objective(name="test_objective", target="max")

    with patch.object(mock_data_access, "get_data", return_value={"test_objective": [1.0]}):
        outcomes = default_evaluation_function(
            uid="123",
            suggestions=suggestions,
            data_access=mock_data_access,
            objectives=[objective],
        )
    assert outcomes == [
        {
            "test_objective": (1.0, None),
            "_id": 1,
        },
    ]


def test_default_evaluation_function_with_baseline(mock_data_access):
    suggestions = [
        {
            "_id": "baseline",
            "x1": 0.0,
            "x2": 0.0,
        },
    ]
    objective = Objective(name="test_objective", target="max")

    with patch.object(mock_data_access, "get_data", return_value={"test_objective": [1.0]}):
        outcomes = default_evaluation_function(
            uid="123",
            suggestions=suggestions,
            data_access=mock_data_access,
            objectives=[objective],
        )
    assert outcomes == [
        {
            "test_objective": (1.0, None),
            "_id": "baseline",
        },
    ]


def test_default_evaluation_function_with_multiple_suggestions(mock_data_access):
    suggestions = [
        {
            "_id": 0,
            "x1": 0.0,
            "x2": 0.0,
        },
        {
            "_id": 1,
            "x1": 1.0,
            "x2": 1.0,
        },
    ]
    objectives = [
        Objective(name="test_objective1", target="max"),
        Objective(name="test_objective2", target="max"),
    ]

    with patch.object(
        mock_data_access, "get_data", return_value={"test_objective1": [1.0, 3.0], "test_objective2": [2.0, 4.0]}
    ):
        outcomes = default_evaluation_function(
            uid="123",
            suggestions=suggestions,
            data_access=mock_data_access,
            objectives=objectives,
        )
    assert outcomes == [
        {
            "test_objective1": (1.0, None),
            "test_objective2": (2.0, None),
            "_id": 0,
        },
        {
            "test_objective1": (3.0, None),
            "test_objective2": (4.0, None),
            "_id": 1,
        },
    ]
    suggestions = [
        {
            "_id": 2,
            "x1": 0.1,
            "x2": 0.1,
        },
        {
            "_id": 3,
            "x1": 1.1,
            "x2": 1.1,
        },
    ]
    with patch.object(
        mock_data_access, "get_data", return_value={"test_objective1": [1.1, 3.1], "test_objective2": [2.1, 4.1]}
    ):
        outcomes = default_evaluation_function(
            uid="124",
            suggestions=suggestions,
            data_access=mock_data_access,
            objectives=objectives,
        )
    assert outcomes == [
        {
            "test_objective1": (1.1, None),
            "test_objective2": (2.1, None),
            "_id": 2,
        },
        {
            "test_objective1": (3.1, None),
            "test_objective2": (4.1, None),
            "_id": 3,
        },
    ]


def test_data_access_evaluation_function(mock_data_access):
    evaluation_function = DataAccessEvaluationFunction(mock_data_access, [Objective(name="test_objective", target="max")])
    suggestions = [
        {
            "_id": 0,
            "x1": 0.0,
            "x2": 0.0,
        },
    ]
    with patch.object(mock_data_access, "get_data", return_value={"test_objective": [1.0]}):
        outcomes = evaluation_function(uid="123", suggestions=suggestions)
    assert outcomes == [
        {
            "test_objective": (1.0, None),
            "_id": 0,
        },
    ]


def test_tiled_evaluation_function():
    tiled_client = MagicMock(spec=Container)
    evaluation_function = TiledEvaluationFunction(tiled_client, [])
    assert isinstance(evaluation_function.data_access, TiledDataAccess)


def test_databroker_evaluation_function():
    broker = MagicMock(spec=Broker)
    evaluation_function = DatabrokerEvaluationFunction(broker, [])
    assert isinstance(evaluation_function.data_access, DatabrokerDataAccess)
