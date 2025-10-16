import asyncio
import logging

import numpy as np
import pytest
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri
from tiled.server.simple import SimpleTiledServer

from blop import DOF, Agent, Objective
from blop.digestion.tests import chankong_and_haimes_digestion, sketchy_himmelblau_digestion
from blop.dofs import BrownianMotion

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

    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    bec.disable_baseline()
    bec.disable_heading()
    bec.disable_table()
    bec.disable_plots()

    return RE


# Agent configuration templates - much more maintainable
AGENT_CONFIGS = {
    "simple_1d": {
        "dofs": [{"name": "x1", "search_domain": (-5.0, 5.0)}],
        "objectives": [{"name": "himmelblau", "target": "min"}],
        "digestion": sketchy_himmelblau_digestion,
    },
    "simple_2d": {
        "dofs": [
            {"name": "x1", "search_domain": (-5.0, 5.0)},
            {"name": "x2", "search_domain": (-5.0, 5.0)},
        ],
        "objectives": [{"name": "himmelblau", "target": "min"}],
        "digestion": sketchy_himmelblau_digestion,
    },
    "constrained_2d": {
        "dofs": [
            {"name": "x1", "search_domain": (-5.0, 5.0)},
            {"name": "x2", "search_domain": (-5.0, 5.0)},
        ],
        "objectives": [
            {"name": "himmelblau", "target": "min"},
            {"name": "himmelblau", "constraint": (95, 105)},
        ],
        "digestion": sketchy_himmelblau_digestion,
    },
    "multiobjective_2d": {
        "dofs": [
            {"name": "x1", "search_domain": (-5.0, 5.0)},
            {"name": "x2", "search_domain": (-5.0, 5.0)},
        ],
        "objectives": [
            {"name": "f1", "target": "min"},
            {"name": "f2", "target": "min"},
            {"name": "c1", "constraint": (-np.inf, 225)},
            {"name": "c2", "constraint": (-np.inf, 0)},
        ],
        "digestion": chankong_and_haimes_digestion,
    },
    "complex_3d": {
        "dofs": [
            {"name": "x1", "search_domain": (-5.0, 5.0)},
            {"name": "x2", "search_domain": (-5.0, 5.0)},
            {"name": "x3", "search_domain": (-5.0, 5.0), "active": False},
            {"device": BrownianMotion(name="brownian1"), "read_only": True},
            {"device": BrownianMotion(name="brownian2"), "read_only": True, "active": False},
        ],
        "objectives": [
            {"name": "himmelblau", "target": "min"},
            {"name": "himmelblau_transpose", "target": "min"},
            {"name": "himmelblau", "constraint": (95, 105)},
        ],
        "digestion": sketchy_himmelblau_digestion,
    },
}


def create_agent_from_config(config_name, db):
    """Create an agent from a configuration template."""
    if config_name not in AGENT_CONFIGS:
        raise ValueError(f"Invalid agent configuration '{config_name}'.")

    config = AGENT_CONFIGS[config_name]

    # Create DOFs
    dofs = []
    for dof_config in config["dofs"]:
        if "device" in dof_config:
            dof = DOF(movable=dof_config["device"], read_only=dof_config.get("read_only", False))
        else:
            dof = DOF(
                description=f"DOF {dof_config['name']}",
                name=dof_config["name"],
                search_domain=dof_config["search_domain"],
                active=dof_config.get("active", True),
            )
        dofs.append(dof)

    # Create objectives
    objectives = []
    for obj_config in config["objectives"]:
        obj = Objective(
            description=obj_config.get("description", obj_config["name"]),
            name=obj_config["name"],
            target=obj_config.get("target"),
            constraint=obj_config.get("constraint"),
        )
        objectives.append(obj)

    return Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=config["digestion"],
        db=db,
    )


# Simplified agent parameter sets - fewer combinations
SIMPLE_AGENTS = ["simple_1d", "simple_2d"]
CONSTRAINED_AGENTS = ["constrained_2d", "multiobjective_2d"]
COMPLEX_AGENTS = ["complex_3d"]
ALL_AGENTS = SIMPLE_AGENTS + CONSTRAINED_AGENTS + COMPLEX_AGENTS


# Focused fixtures for specific testing scenarios
@pytest.fixture(scope="function")
def simple_agent(setup):
    """A simple 2D agent for basic functionality testing."""
    return create_agent_from_config("simple_2d", db=setup)


@pytest.fixture(scope="function")
def constrained_agent(setup):
    """A constrained agent for testing constraint handling."""
    return create_agent_from_config("constrained_2d", db=setup)


@pytest.fixture(scope="function")
def multiobjective_agent(setup):
    """A multi-objective agent for testing Pareto optimization."""
    return create_agent_from_config("multiobjective_2d", db=setup)


@pytest.fixture(scope="function")
def complex_agent(setup):
    """A complex agent with read-only DOFs for advanced testing."""
    return create_agent_from_config("complex_3d", db=setup)


@pytest.fixture
def agent(request, setup):
    agent = create_agent_from_config(request.param, db=setup)

    # add a useless DOF to try and break things
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    return agent
