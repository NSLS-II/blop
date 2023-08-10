import time as ttime

import numpy as np
from ophyd import Signal, SignalRO

DEFAULT_BOUNDS = (-5.0, +5.0)


class DOF(Signal):
    """
    Degree of freedom
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RODOF(DOF):
    """
    Read-only degree of freedom
    """

    ...


class BrownianMotion(RODOF):
    """
    Read-only degree of freedom simulating brownian motion
    """

    def __init__(self, theta=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.theta = theta
        self.old_t = ttime.monotonic()
        self.old_y = 0.0

    def get(self):
        new_t = ttime.monotonic()
        alpha = self.theta ** (new_t - self.old_t)
        new_y = alpha * self.old_y + np.sqrt(1 - alpha**2) * np.random.standard_normal()

        self.old_t = new_t
        self.old_y = new_y
        return new_y


class TimeReadback(SignalRO):
    """
    Returns the current timestamp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConstantReadback(SignalRO):
    """
    Returns a constant every time you read it (more useful than you'd think).
    """

    def __init__(self, constant=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.constant = constant

    def get(self):
        return self.constant
