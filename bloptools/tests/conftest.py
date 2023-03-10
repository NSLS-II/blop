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
