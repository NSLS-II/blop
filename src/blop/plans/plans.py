import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.protocols import Readable, Reading
from bluesky.utils import MsgGenerator, plan

from ..protocols import ID_KEY, Actuator, Checkpointable, OptimizationProblem, Sensor
from .utils import route_suggestions

logger = logging.getLogger(__name__)


def _unpack_for_list_scan(suggestions: list[dict], actuators: Sequence[Actuator]) -> list[Any]:
    """Unpack the actuators and inputs into Bluesky list_scan plan arguments."""
    actuators_and_inputs = {actuator: [suggestion[actuator.name] for suggestion in suggestions] for actuator in actuators}
    unpacked_list = []
    for actuator, values in actuators_and_inputs.items():
        unpacked_list.append(actuator)
        unpacked_list.append(values)

    return unpacked_list


@plan
def default_acquire(
    suggestions: list[dict],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    *,
    per_step: bp.PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    A default plan to acquire data for optimization. Simply a list scan.

    Includes a default metadata key "blop_suggestion_ids" which can be used to identify
    the suggestions that were acquired for each step of the scan.

    Parameters
    ----------
    suggestions: list[dict]
        A list of dictionaries, each containing the parameterization of a point to evaluate.
        The "_id" key is optional and can be used to identify each suggestion. It is suggested
        to add "_id" values to the run metadata for later identification of the acquired data.
    actuators: Sequence[Actuator]
        The actuators to move and the inputs to move them to.
    sensors: Sequence[Sensor]
        The sensors that produce data to evaluate.
    per_step: bp.PerStep | None, optional
        The plan to execute for each step of the scan.
    **kwargs: Any
        Additional keyword arguments to pass to the list_scan plan.

    Returns
    -------
    str
        The UID of the Bluesky run.

    See Also
    --------
    bluesky.plans.list_scan : The Bluesky plan to acquire data.
    """
    if sensors is None:
        sensors = []
    readables = [s for s in sensors if isinstance(s, Readable)]
    if len(readables) != len(sensors):
        logger.warning(f"Some sensors are not readable and will be ignored. Using only the readable sensors: {readables}")

    if len(suggestions) > 1:
        if all(isinstance(actuator, Readable) for actuator in actuators):
            current_position = yield from read(cast(Sequence[Readable], actuators))
        else:
            current_position = None
        suggestions = route_suggestions(suggestions, starting_position=current_position)

    md = {"blop_suggestions": suggestions}
    plan_args = _unpack_for_list_scan(suggestions, actuators)
    return (
        # TODO: fix argument type in bluesky.plans.list_scan
        yield from bp.list_scan(
            readables,
            *plan_args,  # type: ignore[arg-type]
            per_step=per_step,
            md=md,
            **kwargs,
        )
    )


@plan
def optimize_step(
    optimization_problem: OptimizationProblem,
    n_points: int = 1,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[tuple[list[dict], list[dict]]]:
    """
    A single step of the optimization loop.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    n_points : int, optional
        The number of points to suggest.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A tuple containing the suggestions and outcomes of the step.
    """
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    suggestions = optimizer.suggest(n_points)

    uid = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    outcomes = optimization_problem.evaluation_function(uid, suggestions)
    optimizer.ingest(outcomes)

    return suggestions, outcomes


def _maybe_checkpoint(optimizer: Optimizer, checkpoint_interval: int | None, iteration: int) -> None:
    if checkpoint_interval and (iteration + 1) % checkpoint_interval == 0:
        if not isinstance(optimizer, Checkpointable):
            raise ValueError(
                "The optimizer is not checkpointable. Please review your optimizer configuration or implementation."
            )
        optimizer.checkpoint()


@plan
def _read_step(suggestions: list[dict], outcomes: list[dict], readable_cache: dict[str, Readable]) -> MsgGenerator[None]:
    # TODO: Support for manual suggestions where there is no "_id" key?
    suggestion_by_id = {suggestion[ID_KEY]: suggestion.pop(ID_KEY) for suggestion in suggestions}
    outcome_by_id = {outcome[ID_KEY]: outcome.pop(ID_KEY) for outcome in outcomes}

    for key in sorted(suggestion_by_id.keys()):
        for name, value in suggestion_by_id[key].items():
            if name not in readable_cache:
                readable_cache[name] = ReadableSignal(name, initial_value=value, type="array")
            else:
                readable_cache[name].value = value

        for name, value in outcome_by_id[key].items():
            if name not in readable_cache:
                readable_cache[name] = ReadableSignal(name, initial_value=value, type="array")
            else:
                readable_cache[name].value = value
            
        yield from read(list(readable_cache.values()))
        yield from bps.save()


@plan
def optimize(
    optimization_problem: OptimizationProblem,
    iterations: int = 1,
    n_points: int = 1,
    checkpoint_interval: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    A plan to solve the optimization problem.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    iterations : int, optional
        The number of optimization iterations to run.
    n_points : int, optional
        The number of points to suggest per iteration.
    checkpoint_interval : int | None, optional
        The number of iterations between optimizer checkpoints. If None, checkpoints
        will not be saved. Optimizer must implement the
        :class:`blop.protocols.Checkpointable` protocol.
    *args : Any
        Additional positional arguments to pass to the :func:`optimize_step` plan.
    **kwargs : Any
        Additional keyword arguments to pass to the :func:`optimize_step` plan.

    See Also
    --------
    blop.protocols.OptimizationProblem : The problem to solve.
    blop.protocols.Checkpointable : The protocol for checkpointable objects.
    optimize_step : The plan to execute a single step of the optimization.
    """

    # Cache to track readables created from suggestions and outcomes
    readable_cache: dict[str, Readable] = {}

    for i in range(iterations):
        suggestions, outcomes = yield from optimize_step(optimization_problem, n_points, *args, **kwargs)

        # Read the optimization step into the Bluesky and emit events for each suggestion and outcome
        yield from _read_step(suggestions, outcomes, readable_cache)

        _maybe_checkpoint(optimization_problem.optimizer, checkpoint_interval, i)


@plan
def read(readables: Sequence[Readable], **kwargs: Any) -> MsgGenerator[dict[str, Any]]:
    """
    Read the current values of the given readables.

    Parameters
    ----------
    readables : Sequence[Readable]
        The readables to read.

    Returns
    -------
    dict[str, Any]
        A dictionary of the readable names and their current values.
    """
    results = {}
    for readable in readables:
        results[readable.name] = yield from bps.rd(readable, **kwargs)
    return results


def acquire_baseline(
    optimization_problem: OptimizationProblem,
    parameterization: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    Acquire a baseline reading. Useful for relative outcome constraints.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    parameterization : dict[str, Any] | None = None
        Move the DOFs to the given parameterization, if provided.

    See Also
    --------
    default_acquire : The default plan to acquire data.
    """
    actuators = optimization_problem.actuators
    if parameterization is None:
        if all(isinstance(actuator, Readable) for actuator in actuators):
            parameterization = yield from read(cast(Sequence[Readable], actuators))
        else:
            raise ValueError(
                "All actuators must also implement the Readable protocol to acquire a baseline from current positions."
            )
    if ID_KEY not in parameterization:
        parameterization[ID_KEY] = "baseline"
    optimizer = optimization_problem.optimizer
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    uid = yield from acquisition_plan([parameterization], actuators, optimization_problem.sensors, **kwargs)
    outcome = optimization_problem.evaluation_function(uid, [parameterization])[0]
    data = {**outcome, **parameterization}
    optimizer.ingest([data])


@plan
def take_reading_with_background(
    readables: Sequence[Readable],
    name: str = "primary",
    block_beam: Callable[[], MsgGenerator[None]] | None = None,
    unblock_beam: Callable[[], MsgGenerator[None]] | None = None,
) -> MsgGenerator[Mapping[str, Reading]]:
    """
    Takes a reading of the readables while the beam is blocked and then again while the beam is unblocked.

    Parameters
    ----------
    readables: Sequence[Readable]
        The readables to read.
    name: str = "primary"
        The name of the reading.
    block_beam: Callable[[], MsgGenerator[None]] | None = None
        A callable that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]] | None = None
        A callable that unblocks the beam (e.g. by opening a shutter).

    Returns
    -------
    Mapping[str, Reading]
        The readings from the final trigger_and_read operation.
    """
    if block_beam is None or unblock_beam is None:
        raise ValueError("block_beam and unblock_beam plans must be provided.")
    yield from block_beam()
    yield from bps.trigger_and_read(readables, name=f"{name}_background")
    yield from unblock_beam()
    reading = yield from bps.trigger_and_read(readables, name=name)
    return reading


def per_step_background_read(
    block_beam: Callable[[], MsgGenerator[None]], unblock_beam: Callable[[], MsgGenerator[None]]
) -> bp.PerStep:
    """
    Returns a per-step plan function that takes a reading of the readables while the beam is blocked and then
    again while the beam is unblocked.

    Useful for downstream analysis that requires per-step background readings (e.g. background subtraction).

    Parameters
    ----------
    block_beam: Callable[[], MsgGenerator[None]]
        A callable that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]]
        A callable that unblocks the beam (e.g. by opening a shutter).

    See Also
    --------
    bluesky.plans.one_nd_step : The Bluesky plan to execute for each step of the scan.
    """
    take_reading = functools.partial(take_reading_with_background, block_beam=block_beam, unblock_beam=unblock_beam)
    return functools.partial(bps.one_nd_step, take_reading=take_reading)


@plan
def acquire_with_background(
    suggestions: list[dict],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    *,
    block_beam: Callable[[], MsgGenerator[None]],
    unblock_beam: Callable[[], MsgGenerator[None]],
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    A plan to acquire data for optimization with background readings.

    Parameters
    ----------
    suggestions: list[dict]
        A list of dictionaries, each containing the parameterization of a point to evaluate.
        The "_id" key is optional and can be used to identify each suggestion. It is suggested
        to add "_id" values to the run metadata for later identification of the acquired data.
    actuators: Sequence[Actuator]
        The actuators to move to their suggested positions.
    sensors: Sequence[Sensor] | None = None
        The sensors that produce data to evaluate.
    block_beam: Callable[[], MsgGenerator[None]]
        A Bluesky plan that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]]
        A Bluesky plan that unblocks the beam (e.g. by opening a shutter).
    **kwargs: Any
        Additional keyword arguments to pass to the acquisition plan.

    Returns
    -------
    str
        The UID of the Bluesky run.

    See Also
    --------
    acquire : The base plan to acquire data.
    per_step_background_read : The per-step plan to take background readings.
    """
    if sensors is None:
        sensors = []
    per_step = per_step_background_read(block_beam, unblock_beam)
    return (yield from default_acquire(suggestions, actuators, sensors, per_step=per_step, **kwargs))
