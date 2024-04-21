import asyncio
import datetime

import databroker
import numpy as np
import pytest
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

from blop import DOF, Agent, Objective
from blop.dofs import BrownianMotion
from blop.utils import functions


@pytest.fixture(scope="function")
def db():
    """Return a data broker"""
    # MongoDB backend:
    db = Broker.named("temp")  # mongodb backend
    try:
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)
    except Exception:
        pass

    return db


@pytest.fixture(scope="function")
def RE(db):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)
    RE.subscribe(db.insert)

    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)

    bec.disable_baseline()
    bec.disable_heading()
    bec.disable_table()
    bec.disable_plots()

    return RE


@pytest.fixture(scope="function")
def agent(db):
    """
    A simple agent minimizing Himmelblau's function
    """

    dofs = [
        DOF(name="x1", search_domain=(-5.0, 5.0)),
        DOF(name="x2", search_domain=(-5.0, 5.0)),
    ]

    objectives = [Objective(name="himmelblau", target="min")]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=functions.himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def mo_agent(db):
    """
    An agent minimizing two Himmelblau's functions
    """

    def digestion(db, uid):
        products = db[uid].table()

        for index, entry in products.iterrows():
            products.loc[index, "f1"] = functions.himmelblau(entry.x1, entry.x2)
            products.loc[index, "f2"] = functions.himmelblau(entry.x2, entry.x1)

        return products

    dofs = [
        DOF(name="x1", search_domain=(-5.0, 5.0)),
        DOF(name="x2", search_domain=(-5.0, 5.0)),
    ]

    objectives = [Objective(name="f1", target="min"), Objective(name="f2", target="min")]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def constrained_agent(db):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def digestion(db, uid):
        products = db[uid].table()

        for index, entry in products.iterrows():
            products.loc[index, "f1"] = (entry.x1 - 2) ** 2 + (entry.x2 - 1) + 2
            products.loc[index, "f2"] = 9 * entry.x1 - (entry.x2 - 1) + 2
            products.loc[index, "c1"] = entry.x1**2 + entry.x2**2
            products.loc[index, "c2"] = entry.x1 - 3 * entry.x2 + 10

        return products

    dofs = [
        DOF(description="The first DOF", name="x1", search_domain=(-20, 20)),
        DOF(description="The second DOF", name="x2", search_domain=(-20, 20)),
    ]

    objectives = [
        Objective(description="f1", name="f1", target="min"),
        Objective(description="f2", name="f2", target="min"),
        Objective(description="c1", name="c1", target=(-np.inf, 225)),
        Objective(description="c2", name="c2", target=(-np.inf, 0)),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def agent_with_read_only_dofs(db):
    """
    A simple agent minimizing two Himmelblau's functions
    """

    dofs = [
        DOF(name="x1", search_domain=(-5.0, 5.0)),
        DOF(name="x2", search_domain=(-5.0, 5.0)),
        DOF(name="x3", search_domain=(-5.0, 5.0), active=False),
        DOF(device=BrownianMotion(name="brownian1"), read_only=True),
        DOF(device=BrownianMotion(name="brownian2"), read_only=True, active=False),
    ]

    objectives = [
        Objective(name="himmelblau", target="min"),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=functions.himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def make_dirs():
    root_dir = "/tmp/sirepo-bluesky-data"
    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)
