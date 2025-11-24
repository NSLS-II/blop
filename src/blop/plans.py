import functools
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from ax.api.types import TParameterization, TParameterValue
from bluesky.protocols import Movable, NamedMovable, Readable, Reading
from bluesky.utils import Msg, MsgGenerator, plan
from ophyd import Signal  # type: ignore[import-untyped]

from .dofs import DOF
from .protocols import ID_KEY, OptimizationProblem


def _unpack_for_list_scan(suggestions: list[dict], movables: Sequence[NamedMovable]) -> list[NamedMovable | Any]:
    """Unpack the movables and inputs into Bluesky list_scan plan arguments."""
    movables_and_inputs = {movable: [suggestion[movable.name] for suggestion in suggestions] for movable in movables}
    unpacked_list = []
    for movable, values in movables_and_inputs.items():
        unpacked_list.append(movable)
        unpacked_list.append(values)

    return unpacked_list


@plan
def default_acquire(
    suggestions: list[dict],
    movables: Sequence[NamedMovable],
    readables: Sequence[Readable] | None = None,
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
    movables: Sequence[NamedMovable]
        The movables to move and the inputs to move them to.
    readables: Sequence[Readable]
        The readables to trigger and read.
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
    if readables is None:
        readables = []
    md = {"blop_suggestion_ids": [suggestion.get(ID_KEY, None) for suggestion in suggestions]}
    plan_args = _unpack_for_list_scan(suggestions, movables)
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
    movables = optimization_problem.movables
    suggestions = optimizer.suggest(n_points)
    uid = yield from acquisition_plan(suggestions, movables, optimization_problem.readables, *args, **kwargs)
    outcomes = optimization_problem.evaluation_function(uid, suggestions)
    optimizer.ingest(outcomes)


@plan
def optimize(
    optimization_problem: OptimizationProblem,
    iterations: int = 1,
    n_points: int = 1,
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
    """

    for _ in range(iterations):
        yield from optimize_step(optimization_problem, n_points, *args, **kwargs)


@plan
def list_scan_with_delay(*args: Any, delay: float = 0, **kwargs: Any) -> Generator[Msg, None, str]:
    """
    Accepts all the normal 'scan' parameters, plus an optional delay.

    .. deprecated:: v0.8.2
        This plan is deprecated and will be removed in Blop v1.0.0. See documentation how-to-guides for more information.
    """
    warnings.warn(
        "This plan is deprecated and will be removed in Blop v1.0.0. See documentation how-to-guides for more information.",
        DeprecationWarning,
        stacklevel=2,
    )

    def one_nd_step_with_delay(
        detectors: Sequence[Signal], step: Mapping[Movable, Any], pos_cache: Mapping[Movable, Any]
    ) -> Generator[Msg, None, None]:
        "This is a copy of bluesky.plan_stubs.one_nd_step with a sleep added."
        motors = step.keys()
        yield from bps.move_per_step(step, pos_cache)
        yield from bps.sleep(delay)
        yield from bps.trigger_and_read(list(detectors) + list(motors))

    kwargs.setdefault("per_step", one_nd_step_with_delay)
    uid = yield from bp.list_scan(*args, **kwargs)
    return uid


@plan
def default_acquisition_plan(
    dofs: Sequence[DOF], inputs: Mapping[str, Sequence[Any]], dets: Sequence[Signal], **kwargs: Any
) -> Generator[Msg, None, str]:
    """
    Default acquisition plan.

    .. deprecated:: v0.8.2
        This plan is deprecated and will be removed in Blop v1.0.0. See documentation how-to-guides for more information.

    Parameters
    ----------
    x : list of DOFs or DOFList
        A list of DOFs
    inputs: dict
        A dict of a list of inputs per dof, keyed by dof.name
    dets: list
        A list of detectors to trigger
    """
    warnings.warn(
        "This plan is deprecated and will be removed in Blop v1.0.0. See documentation how-to-guides for more information.",
        DeprecationWarning,
        stacklevel=2,
    )
    delay = kwargs.get("delay", 0)
    args = []
    for dof in dofs:
        args.append(dof.movable)
        args.append(inputs[dof.name])

    uid = yield from list_scan_with_delay(dets, *args, delay=delay)
    return uid


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
    movables = optimization_problem.movables
    if parameterization is None:
        if all(isinstance(movable, Readable) for movable in movables):
            parameterization = yield from read(cast(Sequence[Readable], movables))
        else:
            raise ValueError(
                "All movables must also implement the Readable protocol to acquire a baseline from current positions."
            )
    if ID_KEY not in parameterization:
        parameterization[ID_KEY] = "baseline"
    optimizer = optimization_problem.optimizer
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    uid = yield from acquisition_plan([parameterization], movables, optimization_problem.readables, **kwargs)
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
    movables: Sequence[NamedMovable],
    readables: Sequence[Readable] | None = None,
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
    movables: Sequence[NamedMovable]
        The movables to move to their suggested positions.
    readables: Sequence[Readable] | None = None
        The readables that produce data to evaluate.
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
    if readables is None:
        readables = []
    per_step = per_step_background_read(block_beam, unblock_beam)
    return (yield from default_acquire(suggestions, movables, readables, per_step=per_step, **kwargs))


# ===========================================================================================================================
# TODO: Remove when refactoring the Ax agent.
def _unpack_parameters(dofs: dict[str, DOF], parameterizations: list[TParameterization]) -> list[Movable | TParameterValue]:
    """Unpack the parameterizations into Bluesky plan arguments."""
    unpacked_dict = defaultdict(list)
    for parameterization in parameterizations:
        for dof_name in dofs.keys():
            if dof_name in parameterization:
                unpacked_dict[dof_name].append(parameterization[dof_name])
            else:
                raise ValueError(f"Parameter {dof_name} not found in parameterization. Parameterization: {parameterization}")

    unpacked_list = []
    for dof_name, values in unpacked_dict.items():
        unpacked_list.append(dofs[dof_name].movable)
        unpacked_list.append(values)

    return unpacked_list


@plan
def acquire(
    readables: Sequence[Readable],
    dofs: dict[str, DOF],
    trials: dict[int, TParameterization],
    per_step: bp.PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    A plan to acquire data for optimization.

    TODO: Remove when refactoring the Ax agent.

    Parameters
    ----------
    readables: Sequence[Readable]
        The readables to trigger and read.
    dofs: dict[str, DOF]
        A dictionary mapping DOF names to DOFs.
    trials: dict[int, TParameterization]
        A dictionary mapping trial indices to their suggested parameterizations. Typically only a single trial is provided.
    per_step: bp.PerStep | None = None
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
    plan_args = _unpack_parameters(dofs, trials.values())
    return (
        yield from bp.list_scan(
            readables, *plan_args, md={"ax_trial_indices": list(trials.keys())}, per_step=per_step, **kwargs
        )
    )


# ===========================================================================================================================
