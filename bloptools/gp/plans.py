import time as ttime

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401


def take_background(self):
    timeout = 10

    yield from bps.mv(self.shutter.close_cmd, 1)
    yield from bps.sleep(2.0)

    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (self.shutter.status.get() != 1):
        print(f"Shutter not closed, retrying ... (closed_status = {self.shutter.status.get()})")
        yield from bps.sleep(1.0)
        yield from bps.mv(self.shutter.close_cmd, 1)

    yield from bp.count([self.detector])

    yield from bps.mv(self.shutter.open_cmd, 1)
    yield from bps.sleep(2.0)

    timeout = 10
    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (self.shutter.status.get() != 0):
        print(f"Shutter not open, retrying ... (closed_status = {self.shutter.status.get()})")
        yield from bps.sleep(1.0)
        yield from bps.mv(self.shutter.open_cmd, 1)
