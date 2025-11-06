from unittest.mock import MagicMock

from tiled.client.container import Container

from blop.ax.agent import Agent
from blop.dofs import DOF, DOFConstraint
from blop.objectives import Objective

from .conftest import MovableSignal


def test_agent_configure_experiment():
    """Test that the agent can configure an experiment."""
    mock_db = MagicMock(spec=Container)
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=movable1, x2=movable2)
    dof1 = DOF(movable=movable1, search_domain=(0, 10))
    dof2 = DOF(movable=movable2, search_domain=(0, 10))
    objective = Objective(name="test_objective", target="max")
    agent = Agent(readables=[], dofs=[dof1, dof2], objectives=[objective], db=mock_db, dof_constraints=[constraint])
    agent.configure_experiment(name="test_agent", description="Test the Agent")
