from ophyd import Component as Cpt
from ophyd import Device, Signal


def get_dofs(n=2):
    return [Signal(name=f"x{i+1}", value=0) for i in range(n)]


def get_device(name="dofs", n=2):
    components = {}

    for i in range(n):
        components[f"x{i+1}"] = Cpt(Signal, value=i + 1)

    cls = type("DOF", (Device,), components)

    device = cls(name=name)

    return [getattr(device, attr) for attr in device.read_attrs]
