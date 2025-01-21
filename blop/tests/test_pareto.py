import pytest

from .conftest import pareto_agents


@pytest.mark.parametrize("agent", pareto_agents, indirect=True)
def test_pareto(agent, RE, db):
    agent.db = db
    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()


@pytest.mark.parametrize("acqf", ["qnehvi"])
@pytest.mark.parametrize("agent", pareto_agents, indirect=True)
def test_monte_carlo_pareto_acqfs(agent, RE, db, acqf):
    agent.db = db
    RE(agent.learn("qr", n=4))
    RE(agent.learn(acqf, n=2))
    agent.dofs[0].deactivate()
    RE(agent.learn(acqf, n=2))
    getattr(agent, acqf)
