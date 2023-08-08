import time as ttime

from ophyd import Component as Cpt
from ophyd import Device, Signal, SignalRO


def dummy_dof(name):
    return Signal(name=name, value=0.0)


def dummy_dofs(n=2):
    return [dummy_dof(name=f"x{i+1}") for i in range(n)]


def get_dummy_device(name="dofs", n=2):
    components = {}

    for i in range(n):
        components[f"x{i+1}"] = Cpt(Signal, value=i + 1)

    cls = type("DOF", (Device,), components)

    device = cls(name=name)

    return [getattr(device, attr) for attr in device.read_attrs]


class TimeReadback(SignalRO):
    """
    Returns the current timestamp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        return ttime.time()


class ConstantReadback(SignalRO):
    """
    Returns a constant every time you read it (more useful than you'd think).
    """

    def __init__(self, constant=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.constant = constant

    def get(self):
        return self.constant
