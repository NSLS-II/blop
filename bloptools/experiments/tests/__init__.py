import numpy as np
from ophyd import Component as Cpt
from ophyd import Device, Signal


def get_dummy_dofs(n=2):
    return [Signal(name=f"x{i+1}", value=0) for i in range(n)]


def get_dummy_device(name="dofs", n=2):
    components = {}

    for i in range(n):
        components[f"x{i+1}"] = Cpt(Signal, value=i + 1)

    cls = type("DOF", (Device,), components)

    device = cls(name=name)

    return [getattr(device, attr) for attr in device.read_attrs]


def booth(x1, x2):
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def himmelblau(x1, x2):
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def gaussian_beam_waist(x1, x2):
    return np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)


def himmelblau_digestion(db, uid):
    table = db[uid].table()
    products = {"himmelblau": []}

    for index, entry in table.iterrows():
        products["himmelblau"].append(himmelblau(entry.x1, entry.x2))

    return products


def mock_kbs_digestion(db, uid):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
    """

    table = db[uid].table()
    products = {"x_width": [], "y_width": []}

    for index, entry in table.iterrows():
        sigma_x = gaussian_beam_waist(entry.x1, entry.x2)
        sigma_y = gaussian_beam_waist(entry.x3, entry.x4)

        products["x_width"].append(2 * sigma_x)
        products["y_width"].append(2 * sigma_y)

    return products
