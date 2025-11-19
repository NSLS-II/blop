import asyncio

import pytest
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker

from blop.data_access import DatabrokerDataAccess, TiledDataAccess

from .conftest import create_agent_from_config


@pytest.fixture(scope="function", params=["databroker", "tiled"])
def backend_setup(request):
    """Set up backend for data access tests."""
    backend = request.param

    if backend == "databroker":
        import databroker
        from databroker import Broker

        from blop.sim import HDF5Handler

        db = Broker.named("temp")
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        yield db

    elif backend == "tiled":
        from tiled.client import from_uri
        from tiled.server.simple import SimpleTiledServer

        server = SimpleTiledServer(readable_storage=["/tmp/blop/sim"])
        client = from_uri(server.uri)
        yield client
        server.close()


@pytest.fixture(scope="function")
def db_callback_backend(backend_setup):
    """Returns the TiledWriter callback for the default tiled backend."""
    if isinstance(backend_setup, Broker):
        return backend_setup.insert, backend_setup
    else:
        return TiledWriter(backend_setup), backend_setup


@pytest.fixture(scope="function")
def RE_backend(db_callback_backend):
    """Sets up the RunEngine with the correct callback."""
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)

    RE.subscribe(db_callback_backend[0])

    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    bec.disable_baseline()
    bec.disable_heading()
    bec.disable_table()
    bec.disable_plots()

    return RE, db_callback_backend[1]


def test_agent_data_access(RE_backend):
    """Test that the original agent can work with different backends."""
    RE = RE_backend[0]
    db = RE_backend[1]
    # Create a simple agent
    agent = create_agent_from_config("simple_2d", db=db)

    # Verify agent was created correctly
    assert len(agent.dofs) == 2
    assert len(agent.objectives) == 1
    if isinstance(db, Broker):
        assert isinstance(agent.data_access, DatabrokerDataAccess)
    else:
        assert isinstance(agent.data_access, TiledDataAccess)

    # Learn a bit
    RE(agent.learn("qr", n=16))
    RE(agent.learn("qei", n=2))
