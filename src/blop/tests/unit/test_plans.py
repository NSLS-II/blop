from unittest.mock import MagicMock, Mock, patch

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine

from blop.plans import acquire_baseline, acquire_with_background, default_acquire, optimize, optimize_step
from blop.protocols import AcquisitionPlan, EvaluationFunction, OptimizationProblem, Optimizer

from .conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_optimize(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[MovableSignal("x1", initial_value=-1.0)],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(optimize(optimization_problem))

    optimizer.suggest.assert_called_once_with(1)
    optimizer.ingest.assert_called_once_with([{"objective": 0.0}])
    assert evaluation_function.call_count == 1


def test_optimize_multiple(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[MovableSignal("x1", initial_value=-1.0)],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(optimize(optimization_problem, iterations=5))

    optimizer.suggest.assert_called_with(1)
    optimizer.ingest.assert_called_with([{"objective": 0.0}])
    assert optimizer.suggest.call_count == 5
    assert optimizer.ingest.call_count == 5
    assert evaluation_function.call_count == 5


def test_optimize_multiple_with_n_points(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}, {"x1": 0.1, "_id": 1}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0}, {"objective": 0.1}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[MovableSignal("x1", initial_value=-1.0)],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )
    RE(optimize(optimization_problem, iterations=5, n_points=2))
    optimizer.suggest.assert_called_with(2)
    optimizer.ingest.assert_called_with([{"objective": 0.0}, {"objective": 0.1}])
    assert optimizer.suggest.call_count == 5
    assert optimizer.ingest.call_count == 5
    assert evaluation_function.call_count == 5


def test_optimize_complex_case(RE):
    """Test with multi-suggest, multi-parameter, multi-objective, multi-readable case."""
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "_id": 0},
        {"x1": 0.1, "x2": 0.2, "x3": 0.3, "_id": 1},
    ]
    evaluation_function = MagicMock(
        spec=EvaluationFunction,
        return_value=[
            {"objective1": 0.0, "objective2": 0.1},
            {"objective1": 0.1, "objective2": 0.2},
        ],
    )
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[
            MovableSignal("x1", initial_value=-1.0),
            MovableSignal("x2", initial_value=-1.0),
            MovableSignal("x3", initial_value=-1.0),
        ],
        readables=[ReadableSignal("readable1"), ReadableSignal("readable2")],
        evaluation_function=evaluation_function,
    )

    RE(optimize(optimization_problem, iterations=2, n_points=2))

    optimizer.suggest.assert_called_with(2)
    optimizer.ingest.assert_called_with([{"objective1": 0.0, "objective2": 0.1}, {"objective1": 0.1, "objective2": 0.2}])
    assert optimizer.suggest.call_count == 2
    assert optimizer.ingest.call_count == 2
    assert evaluation_function.call_count == 2


def test_optimize_step_default(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[MovableSignal("x1", initial_value=-1.0)],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(optimize_step(optimization_problem))

    optimizer.suggest.assert_called_once_with(1)
    optimizer.ingest.assert_called_once_with([{"objective": 0.0}])
    assert evaluation_function.call_count == 1


def test_optimize_step_custom_acquisition_plan(RE):
    acquisition_plan = MagicMock(spec=AcquisitionPlan)
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[movable],
        readables=[readable],
        evaluation_function=evaluation_function,
        acquisition_plan=acquisition_plan,
    )

    RE(optimize_step(optimization_problem))
    optimizer.suggest.assert_called_once_with(1)
    acquisition_plan.assert_called_once_with(
        [{"x1": 0.0, "_id": 0}],
        [movable],
        [readable],
    )
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}])
    assert evaluation_function.call_count == 1


def test_default_acquire_single_movable_readable(RE):
    """Test with single movable, position, and readable."""
    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            default_acquire(
                [{"x1": 0.0, "_id": 0}],
                [movable],
                [readable],
            )
        )
        assert mock_read.call_count == 1

    assert movable.read()["x1"]["value"] == 0.0


def test_default_acquire_multiple_movables_readables(RE):
    """Test with multiple movables, positions, and readables."""
    movable1 = MovableSignal("x1", initial_value=-1.0)
    movable2 = MovableSignal("x2", initial_value=-1.0)
    readable1 = ReadableSignal("objective1")
    readable2 = ReadableSignal("objective2")

    with (
        patch.object(movable1, "set", wraps=movable1.set) as mock_set1,
        patch.object(movable2, "set", wraps=movable2.set) as mock_set2,
        patch.object(readable1, "read", wraps=readable1.read) as mock_read1,
        patch.object(readable2, "read", wraps=readable2.read) as mock_read2,
    ):
        RE(
            default_acquire(
                [{"x1": 0.0, "x2": 0.0, "_id": 0}, {"x1": 0.1, "x2": 0.1, "_id": 1}],
                [movable1, movable2],
                [readable1, readable2],
            )
        )

        # Verify movables were set in correct order
        assert mock_set1.call_count == 2
        assert mock_set2.call_count == 2
        assert mock_set1.call_args_list[0][0][0] == 0.0  # First call
        assert mock_set2.call_args_list[0][0][0] == 0.0
        assert mock_set1.call_args_list[1][0][0] == 0.1  # Second call
        assert mock_set2.call_args_list[1][0][0] == 0.1

        # Verify reads happened twice
        assert mock_read1.call_count == 2
        assert mock_read2.call_count == 2

    # Verify final positions
    assert movable1.read()["x1"]["value"] == 0.1
    assert movable2.read()["x2"]["value"] == 0.1


def test_acquire_with_background(RE):
    """Test background acquisition with multiple movables, positions, and readables"""

    def block_beam():
        yield from bps.null()

    def unblock_beam():
        yield from bps.null()

    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")

    mock_block_beam = Mock(wraps=block_beam)
    mock_unblock_beam = Mock(wraps=unblock_beam)

    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            acquire_with_background(
                [{"x1": 0.0, "_id": 0}],
                [movable],
                readables=[readable],
                block_beam=mock_block_beam,
                unblock_beam=mock_unblock_beam,
            )
        )
        # Two reads, one blocked, one unblocked
        assert mock_read.call_count == 2
        assert mock_block_beam.call_count == 1
        assert mock_unblock_beam.call_count == 1

    assert movable.read()["x1"]["value"] == 0.0


def test_acquire_baseline(RE):
    """Test acquiring a baseline reading from suggested parameterizations."""
    optimizer = MagicMock(spec=Optimizer)
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": "baseline"}])

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[MovableSignal("x1", initial_value=-1.0)],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(acquire_baseline(optimization_problem, parameterization={"x1": 0.0}))

    # No suggestions are made since this is a baseline reading
    assert optimizer.suggest.call_count == 0

    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": "baseline", "x1": 0.0}])
    assert evaluation_function.call_count == 1


def test_acquire_baseline_from_current(RE):
    """Test acquiring a baseline reading from the current movable positions."""
    optimizer = MagicMock(spec=Optimizer)
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": "baseline"}])
    movable = MovableSignal("x1", initial_value=-1.0)

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        movables=[movable],
        readables=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    with (
        patch.object(movable, "set", wraps=movable.set) as mock_set,
        patch.object(movable, "read", wraps=movable.read) as mock_read,
    ):
        RE(acquire_baseline(optimization_problem))

        # Ensure the movable was read twice (once for the baseline, once during the acquisition)
        assert mock_read.call_count == 2
        # Ensure the movable was set once to the current value
        assert mock_set.call_count == 1
        assert mock_set.call_args_list[0][0][0] == -1.0

    # No suggestions are made since this is a baseline reading from the current movable positions
    assert optimizer.suggest.call_count == 0

    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": "baseline", "x1": -1.0}])
    assert evaluation_function.call_count == 1
