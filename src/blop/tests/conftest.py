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
def agent_1d_1f(db):
    """
    A one dimensional agent.
    """

    def digestion(df):
        for index, entry in df.iterrows():
            df.loc[index, "f1"] = functions.himmelblau(entry.x1, 3)

        return df

    dofs = DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0))
    objectives = Objective(description="f1", name="f1", target="min")

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=digestion,
        db=db,
    )

    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), read_only=False))

    return agent


@pytest.fixture(scope="function")
def agent_2d_1f(db):
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

    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), read_only=False))

    return agent


@pytest.fixture(scope="function")
def agent_2d_2f(db):
    """
    An agent minimizing two Himmelblau's functions
    """

    def digestion(df):
        for index, entry in df.iterrows():
            df.loc[index, "f1"] = functions.himmelblau(entry.x1, entry.x2)
            df.loc[index, "f2"] = functions.himmelblau(entry.x2, entry.x1)

        return df

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

    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), read_only=False))

    return agent


@pytest.fixture(scope="function")
def agent_2d_2f_2c(db):
    """
    Chankong and Haimes function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def digestion(df):
        for index, entry in df.iterrows():
            df.loc[index, "f1"] = (entry.x1 - 2) ** 2 + (entry.x2 - 1) + 2
            df.loc[index, "f2"] = 9 * entry.x1 - (entry.x2 - 1) + 2
            df.loc[index, "c1"] = entry.x1**2 + entry.x2**2
            df.loc[index, "c2"] = entry.x1 - 3 * entry.x2 + 10

        return df

    dofs = [
        DOF(description="The first DOF", name="x1", search_domain=(-20, 20), travel_expense=1.0),
        DOF(description="The second DOF", name="x2", search_domain=(-20, 20), travel_expense=2.0),
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

    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), read_only=False))

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

    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), read_only=False))

    return agent


@pytest.fixture(scope="function")
def make_dirs():
    root_dir = "/tmp/sirepo-bluesky-data"
    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)
