import logging
from collections.abc import Sequence
from typing import Any, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.protocols import Readable
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
) -> MsgGenerator[dict]:
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
    dict
        Metadata describing the acquisition. Contains the UID of the Bluesky run.

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
    # TODO: fix argument type in bluesky.plans.list_scan
    uid = yield from bp.list_scan(
        readables,
        *plan_args,  # type: ignore[arg-type]
        per_step=per_step,
        md=md,
        **kwargs,
    )

    return {"uid": uid}


@plan
def optimize_step(
    optimization_problem: OptimizationProblem,
    n_points: int = 1,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    A single step of the optimization loop.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    n_points : int, optional
        The number of points to suggest.
    """
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    suggestions = optimizer.suggest(n_points)

    acquisition_md = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    outcomes = optimization_problem.evaluation_function(acquisition_md, suggestions)
    optimizer.ingest(outcomes)


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

    for i in range(iterations):
        yield from optimize_step(optimization_problem, n_points, *args, **kwargs)
        if checkpoint_interval and (i + 1) % checkpoint_interval == 0:
            if not isinstance(optimization_problem.optimizer, Checkpointable):
                raise ValueError(
                    "The optimizer is not checkpointable. Please review your optimizer configuration or implementation."
                )
            optimization_problem.optimizer.checkpoint()


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
    acquisition_md = yield from acquisition_plan([parameterization], actuators, optimization_problem.sensors, **kwargs)
    outcome = optimization_problem.evaluation_function(acquisition_md, [parameterization])[0]
    data = {**outcome, **parameterization}
    optimizer.ingest([data])
