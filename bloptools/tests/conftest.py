import asyncio
import datetime

import databroker
import pytest
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree
from sirepo_bluesky.madx_handler import MADXFileHandler
from sirepo_bluesky.shadow_handler import ShadowFileHandler
from sirepo_bluesky.sirepo_bluesky import SirepoBluesky
from sirepo_bluesky.srw_handler import SRWFileHandler

from bloptools.bayesian import Agent

from .. import devices, test_functions


@pytest.fixture(scope="function")
def db():
    """Return a data broker"""
    # MongoDB backend:
    db = Broker.named("local")  # mongodb backend
    try:
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)
    except Exception:
        pass

    db.reg.register_handler("srw", SRWFileHandler, overwrite=True)
    db.reg.register_handler("shadow", ShadowFileHandler, overwrite=True)
    db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)
    db.reg.register_handler("madx", MADXFileHandler, overwrite=True)

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
        {"device": devices.DOF(name="x1"), "limits": (-5, 5), "kind": "active"},
        {"device": devices.DOF(name="x2"), "limits": (-5, 5), "kind": "active"},
    ]

    tasks = [
        {"key": "himmelblau", "kind": "minimize"},
    ]

    agent = Agent(
        dofs=dofs,
        tasks=tasks,
        digestion=test_functions.himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    return agent


@pytest.fixture(scope="function")
def multitask_agent(db):
    """
    A simple agent minimizing two Styblinski-Tang functions
    """

    def digestion(db, uid):
        products = db[uid].table()

        for index, entry in products.iterrows():
            products.loc[index, "ST1"] = test_functions.styblinski_tang(entry.x1, entry.x2)
            products.loc[index, "ST2"] = test_functions.styblinski_tang(entry.x1, -entry.x2)

        return products

    dofs = [
        {"device": devices.DOF(name="x1"), "limits": (-5, 5), "kind": "active"},
        {"device": devices.DOF(name="x2"), "limits": (-5, 5), "kind": "active"},
    ]

    tasks = [
        {"key": "ST1", "kind": "minimize"},
        {"key": "ST2", "kind": "minimize"},
    ]

    agent = Agent(
        dofs=dofs,
        tasks=tasks,
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


@pytest.fixture(scope="function")
def srw_tes_simulation(make_dirs):
    connection = SirepoBluesky("http://localhost:8000")
    data, _ = connection.auth("srw", "00000002")
    return connection


@pytest.fixture(scope="function")
def shadow_tes_simulation(make_dirs):
    connection = SirepoBluesky("http://localhost:8000")
    data, _ = connection.auth("shadow", "00000002")
    return connection
