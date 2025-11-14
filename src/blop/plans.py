import functools
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from ax.api.types import TParameterization, TParameterValue
from bluesky.protocols import Movable, Readable, Reading
from bluesky.run_engine import Msg
from bluesky.utils import MsgGenerator, plan
from ophyd import Signal  # type: ignore[import-untyped]

from .dofs import DOF
from .ax.agent import Agent


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


@plan
def optimize_step(
    generator: Agent,
    n: int = 1,
    acquisition_plan: Callable[[], MsgGenerator[None]] | None = None,
) -> MsgGenerator[None]:
    """
    A single step of the optimization loop.

    Parameters
    ----------
    generator : Agent
        The generator to optimize with.
    n : int, optional
        The number of trials to suggest.
    acquisition_plan : Callable[[], MsgGenerator[None]] | None, optional
        The acquisition plan to use to acquire data. If not provided, the default acquisition plan will be used.
    """
    if acquisition_plan is None:
        acquisition_plan = acquire
    trials = generator.suggest(n)
    data = yield from acquisition_plan(generator.readables, generator.dofs, trials)
    outcomes = generator.evaluate(trials, data)
    generator.ingest(outcomes)


@plan
def optimize(
    generator: Agent,
    iterations: int = 1,
    n: int = 1,
    acquisition_plan: Callable[[], MsgGenerator[None]] | None = None,
) -> MsgGenerator[None]:
    """
    A plan to optimize the generator.

    Parameters
    ----------
    generator : Agent
        The generator to optimize with.
    iterations : int, optional
        The number of optimization iterations to run.
    n : int, optional
        The number of trials to suggest per iteration.
    acquisition_plan : Callable[[], MsgGenerator[None]] | None, optional
        The acquisition plan to use to acquire data. If not provided, the default acquisition plan will be used.
    """

    for _ in range(iterations):
        yield from optimize_step(generator, n, acquisition_plan)


@plan
def list_scan_with_delay(*args: Any, delay: float = 0, **kwargs: Any) -> Generator[Msg, None, str]:
    "Accepts all the normal 'scan' parameters, plus an optional delay."

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
    Parameters
    ----------
    x : list of DOFs or DOFList
        A list of DOFs
    inputs: dict
        A dict of a list of inputs per dof, keyed by dof.name
    dets: list
        A list of detectors to trigger
    """
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
    """
    results = {}
    for readable in readables:
        results[readable.name] = yield from bps.rd(readable, **kwargs)
    return results


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
    readables: Sequence[Readable],
    dofs: Sequence[DOF],
    trials: dict[int, TParameterization],
    block_beam: Callable[[], MsgGenerator[None]],
    unblock_beam: Callable[[], MsgGenerator[None]],
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    A plan to acquire data for optimization with background readings.

    Parameters
    ----------
    readables: Sequence[Readable]
        The readables to trigger and read.
    dofs: Sequence[DOF]
        The DOFs to move.
    trials: dict[int, TParameterization]
        A dictionary mapping trial indices to their suggested parameterizations. Typically only a single trial is provided.
    block_beam: Callable[[], MsgGenerator[None]]
        A callable that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]]
        A callable that unblocks the beam (e.g. by opening a shutter).
    **kwargs: Any
        Additional keyword arguments to pass to the list_scan plan.

    Returns
    -------
    str
        The UID of the Bluesky run.

    See Also
    --------
    acquire : The base plan to acquire data.
    per_step_background_read : The per-step plan to take background readings.
    """
    per_step = per_step_background_read(block_beam, unblock_beam)
    return (yield from acquire(readables, dofs, trials, per_step=per_step, **kwargs))


def acquire_baseline(
    generator: Agent,
    parameterization: TParameterization | None = None,
    arm_name: str | None = None,
    acquisition_plan: Callable[[Sequence[Readable], dict[str, DOF], dict[int, TParameterization], Any], MsgGenerator[str]] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    Acquire a baseline reading.

    Parameters
    ----------
    generator: Agent
        The generator to acquire the baseline for.
    parameterization : TParameterization, optional
        Move the DOFs to the given parameterization, if provided.
    arm_name : str, optional
        A name for the arm to distinguish it from other arms.
    per_step: bp.PerStep | None, optional
        The per-step plan to execute for each step of the scan.
    **kwargs: Any
        Additional keyword arguments to pass to the acquire plan.
    """
    if parameterization is None:
        parameterization = yield from read([dof.movable for dof in generator.dofs.values()])
    trials = generator.attach_baseline(parameters=parameterization, arm_name=arm_name)
    uid = yield from acquisition_plan(generator.readables, generator.dofs, trials, **kwargs)
    outcomes = generator.evaluation_function(trials, uid, **generator.evaluation_kwargs)
    generator.ingest(trials, outcomes)