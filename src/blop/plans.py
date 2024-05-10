import bluesky.plan_stubs as bps
import bluesky.plans as bp


def list_scan_with_delay(*args, delay=0, **kwargs):
    "Accepts all the normal 'scan' parameters, plus an optional delay."

    def one_nd_step_with_delay(detectors, step, pos_cache):
        "This is a copy of bluesky.plan_stubs.one_nd_step with a sleep added."
        motors = step.keys()
        yield from bps.move_per_step(step, pos_cache)
        yield from bps.sleep(delay)
        yield from bps.trigger_and_read(list(detectors) + list(motors))

    kwargs.setdefault("per_step", one_nd_step_with_delay)
    uid = yield from bp.list_scan(*args, **kwargs)
    return uid


def default_acquisition_plan(dofs, inputs, dets, **kwargs):
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
        args.append(dof.device)
        args.append(inputs[dof.name])

    uid = yield from list_scan_with_delay(dets, *args, delay=delay)
    return uid
