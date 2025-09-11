import asyncio
import logging

import databroker  # type: ignore[import-untyped]
import numpy as np
import pytest
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from tiled.client import from_uri
from tiled.server.simple import SimpleTiledServer

from blop import DOF, Agent, Objective
from blop.digestion.tests import chankong_and_haimes_digestion, sketchy_himmelblau_digestion
from blop.dofs import BrownianMotion
from blop.sim import HDF5Handler

logger = logging.getLogger("blop")
logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="function", params=["databroker", "tiled"])
def backend(request):
    """Parameterizes tests to run for databroker and tiled."""
    return request.param


@pytest.fixture(scope="function")
def setup(backend):
    """Returns the database or client object based on the backend."""
    if backend == "databroker":
        db = Broker.named("temp")
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        yield db

    elif backend == "tiled":
        server = SimpleTiledServer(readable_storage=["/tmp/blop/sim"])
        client = from_uri(server.uri)
        yield client
        server.close()

    else:
        pytest.fail(f"Invalid backend specified: {backend}")


@pytest.fixture(scope="function")
def db_callback(backend, setup):
    """Returns the callback function based on the backend."""
    if backend == "databroker":
        return setup.insert

    elif backend == "tiled":
        return TiledWriter(setup)


@pytest.fixture(scope="function")
def RE(db_callback):
    """Sets up the RunEngine with the correct callback."""
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)

    RE.subscribe(db_callback)

    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    bec.disable_baseline()
    bec.disable_heading()
    bec.disable_table()
    bec.disable_plots()

    return RE


single_task_agents = [
    "1d_1f",
    "2d_1f",
    "2d_1f_1c",
    "2d_2f_2c",
    "3d_2r_2f_1c",
]

nonpareto_multitask_agents = ["2d_2c"]

pareto_agents = ["2d_2f_2c", "3d_2r_2f_1c"]

all_agents = [*single_task_agents, *nonpareto_multitask_agents, *pareto_agents]


def get_agent(param, db):
    """
    Generate a bunch of different agents.
    """
    if param == "1d_1f":
        return Agent(
            dofs=[DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0))],
            objectives=[Objective(description="Himmelblau’s function", name="himmelblau", target="min")],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    elif param == "1d_1c":
        return Agent(
            dofs=[DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0))],
            objectives=[Objective(description="Himmelblau’s function", name="himmelblau", constraint=(95, 105))],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    elif param == "2d_1f":
        return Agent(
            dofs=[
                DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0)),
                DOF(description="The first DOF", name="x2", search_domain=(-5.0, 5.0)),
            ],
            objectives=[Objective(description="Himmelblau’s function", name="himmelblau", target="min")],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    elif param == "2d_2c":
        return Agent(
            dofs=[
                DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0)),
                DOF(description="The first DOF", name="x2", search_domain=(-5.0, 5.0)),
            ],
            objectives=[
                Objective(description="Himmelblau’s function", name="himmelblau", constraint=(95, 105)),
                Objective(description="Himmelblau’s function", name="himmelblau", constraint=(95, 105)),
            ],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    elif param == "2d_1f_1c":
        return Agent(
            dofs=[
                DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0)),
                DOF(description="The first DOF", name="x2", search_domain=(-5.0, 5.0)),
            ],
            objectives=[
                Objective(description="Himmelblau’s function", name="himmelblau", target="min"),
                Objective(description="Himmelblau’s function", name="himmelblau", constraint=(95, 105)),
            ],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    elif param == "2d_2f_2c":
        return Agent(
            dofs=[
                DOF(description="The first DOF", name="x1", search_domain=(-5.0, 5.0)),
                DOF(description="The first DOF", name="x2", search_domain=(-5.0, 5.0)),
            ],
            objectives=[
                Objective(description="f1", name="f1", target="min"),
                Objective(description="f2", name="f2", target="min"),
                Objective(description="c1", name="c1", constraint=(-np.inf, 225)),
                Objective(description="c2", name="c2", constraint=(-np.inf, 0)),
            ],
            digestion=chankong_and_haimes_digestion,
            db=db,
        )

    elif param == "3d_2r_2f_1c":
        return Agent(
            dofs=[
                DOF(name="x1", search_domain=(-5.0, 5.0)),
                DOF(name="x2", search_domain=(-5.0, 5.0)),
                DOF(name="x3", search_domain=(-5.0, 5.0), active=False),
                DOF(device=BrownianMotion(name="brownian1"), read_only=True),
                DOF(device=BrownianMotion(name="brownian2"), read_only=True, active=False),
            ],
            objectives=[
                Objective(name="himmelblau", target="min"),
                Objective(name="himmelblau_transpose", target="min"),
                Objective(description="Himmelblau’s function", name="himmelblau", constraint=(95, 105)),
            ],
            digestion=sketchy_himmelblau_digestion,
            db=db,
        )

    else:
        raise ValueError(f"Invalid agent parameter '{param}'.")


@pytest.fixture
def agent(request, setup):
    agent = get_agent(request.param, db=setup)

    # add a useless DOF to try and break things
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    return agent
