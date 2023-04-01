import time as ttime

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401


def take_background(shutter, detectors):
    timeout = 10

    yield from bps.mv(shutter.close_cmd, 1)
    yield from bps.sleep(2.0)

    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (shutter.status.get(as_string=True) == "Open"):
        print(f"Shutter not closed, retrying ... (closed_status = {shutter.status.get(as_string=True)})")
        yield from bps.sleep(1.0)
        yield from bps.mv(shutter.close_cmd, 1)

    uid = yield from bp.count(detectors)

    yield from bps.mv(shutter.open_cmd, 1)
    yield from bps.sleep(2.0)

    timeout = 10
    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (shutter.status.get(as_string=True) != "Open"):
        print(f"Shutter not open, retrying ... (closed_status = {shutter.status.get(as_string=True)})")
        yield from bps.sleep(1.0)
        yield from bps.mv(shutter.open_cmd, 1)

    yield from bps.sleep(10.0)

    return uid
