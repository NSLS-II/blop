import asyncio
import logging

import pytest
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri
from tiled.server.simple import SimpleTiledServer

logger = logging.getLogger("blop")
logger.setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(scope="function")
def setup():
    """Returns the tiled client as the default backend for all tests."""
    server = SimpleTiledServer(readable_storage=["/tmp/blop/sim"])
    client = from_uri(server.uri)
    yield client
    server.close()


@pytest.fixture(scope="function")
def db_callback(setup):
    """Returns the TiledWriter callback for the default tiled backend."""
    return TiledWriter(setup)


@pytest.fixture(scope="function")
def RE(db_callback):
    """Sets up the RunEngine with the correct callback."""
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)
    RE.subscribe(db_callback)
    return RE
