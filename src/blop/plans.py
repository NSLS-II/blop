import functools
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from ax.api.types import TParameterization, TParameterValue
from bluesky.protocols import Movable, Readable, Reading
from bluesky.run_engine import Msg
from bluesky.utils import MsgGenerator
from ophyd import Signal  # type: ignore[import-untyped]

from .dofs import DOF


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
        Only the reading with the given name is returned.
    """
    if block_beam is None or unblock_beam is None:
        raise ValueError("block_beam and unblock_beam plans must be provided.")
    yield from block_beam()
    yield from bps.trigger_and_read(readables, name=f"{name}_background")
    yield from unblock_beam()
    return (yield from bps.trigger_and_read(readables, name=name))


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

    def take_reading(readables: Sequence[Readable], name: str = "primary") -> MsgGenerator[Mapping[str, Reading]]:
        yield from block_beam()
        yield from bps.trigger_and_read(readables, name=f"{name}_background")
        yield from unblock_beam()
        yield from bps.trigger_and_read(readables, name=name)

    return functools.partial(bps.one_nd_step, take_reading=take_reading)


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


def acquire(
    readables: Sequence[Readable],
    dofs: Sequence[DOF],
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
    dofs: Sequence[DOF]
        The DOFs to move.
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
    uid = yield from bp.list_scan(
        readables, *plan_args, md={"ax_trial_indices": list(trials.keys())}, per_step=per_step, **kwargs
    )
    return uid
