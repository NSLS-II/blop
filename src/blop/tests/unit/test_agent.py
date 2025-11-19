from blop.ax.agent import Agent
from blop.dofs import DOF, DOFConstraint
from blop.objectives import Objective

from .conftest import MovableSignal, ReadableSignal


def test_agent_configure_experiment():
    """Test that the agent can configure an experiment."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=movable1, x2=movable2)
    dof1 = DOF(movable=movable1, search_domain=(0, 10))
    dof2 = DOF(movable=movable2, search_domain=(0, 10))
    objective = Objective(name="test_objective", target="max")
    agent = Agent(readables=[], dofs=[dof1, dof2], objectives=[objective], dof_constraints=[constraint])
    agent.configure_experiment(name="test_agent", description="Test the Agent")


def test_agent_to_optimization_problem():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = DOF(movable=movable1, search_domain=(0, 10))
    dof2 = DOF(movable=movable2, search_domain=(0, 10))
    readable = ReadableSignal(name="test_readable")
    objective = Objective(name="test_objective", target="max")
    agent = Agent(readables=[readable], dofs=[dof1, dof2], objectives=[objective])

    optimization_problem = agent.to_optimization_problem()
    assert optimization_problem.generator == agent
    assert optimization_problem.movables == [movable1, movable2]
    assert optimization_problem.readables == [readable]
    assert optimization_problem.evaluation_function == agent.evaluation_function
    assert optimization_problem.acquisition_plan is None


def test_agent_suggest():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = DOF(movable=movable1, search_domain=(0, 10))
    dof2 = DOF(movable=movable2, search_domain=(0, 10))
    objective = Objective(name="test_objective", target="max")
    agent = Agent(readables=[], dofs=[dof1, dof2], objectives=[objective])
    agent.configure_experiment(name="test_agent", description="Test the Agent")

    parameterizations = agent.suggest(1)
    assert len(parameterizations) == 1
    assert parameterizations[0]["_id"] == 0
    assert "test_movable1" in parameterizations[0]
    assert "test_movable2" in parameterizations[0]
    assert isinstance(parameterizations[0]["test_movable1"], (int, float))
    assert isinstance(parameterizations[0]["test_movable2"], (int, float))


def test_agent_suggest_multiple():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = DOF(movable=movable1, search_domain=(0, 10))
    dof2 = DOF(movable=movable2, search_domain=(0, 10))
    objective = Objective(name="test_objective", target="max")
    agent = Agent(readables=[], dofs=[dof1, dof2], objectives=[objective])
    agent.configure_experiment(name="test_agent", description="Test the Agent")

    parameterizations = agent.suggest(5)
    assert len(parameterizations) == 5
    for i in range(5):
        assert parameterizations[i]["_id"] == i
        assert "test_movable1" in parameterizations[i]
        assert "test_movable2" in parameterizations[i]
        assert isinstance(parameterizations[i]["test_movable1"], (int, float))
        assert isinstance(parameterizations[i]["test_movable2"], (int, float))
