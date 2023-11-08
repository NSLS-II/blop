import asyncio
import datetime

import databroker
import pytest
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

from bloptools.bayesian import DOF, Agent, Objective
from bloptools.utils import functions


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
        DOF(name="x1", limits=(-8.0, 8.0)),
        DOF(name="x2", limits=(-8.0, 8.0)),
    ]

    objectives = [Objective(key="himmelblau", minimize=True)]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=functions.constrained_himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def multi_agent(db):
    """
    A simple agent minimizing two Styblinski-Tang functions
    """

    def digestion(db, uid):
        products = db[uid].table()

        for index, entry in products.iterrows():
            products.loc[index, "ST1"] = functions.styblinski_tang(entry.x1, entry.x2)
            products.loc[index, "ST2"] = functions.styblinski_tang(entry.x1, -entry.x2)

        return products

    dofs = [
        DOF(name="x1", limits=(-5.0, 5.0)),
        DOF(name="x2", limits=(-5.0, 5.0)),
    ]

    objectives = [Objective(key="ST1", minimize=True), Objective(key="ST2", minimize=True)]

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
def make_dirs():
    root_dir = "/tmp/sirepo-bluesky-data"
    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)
