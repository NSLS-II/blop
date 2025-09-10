import pytest

from .conftest import all_agents


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
@pytest.mark.parametrize("agent", all_agents, indirect=True)
def test_analytic_acqfs(agent, RE, acqf):
    RE(agent.learn("qr", n=32))
    RE(agent.learn(acqf, n=1))
    getattr(agent, acqf)


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
@pytest.mark.parametrize("agent", all_agents, indirect=True)
def test_monte_carlo_acqfs(agent, RE, acqf):
    RE(agent.learn("qr", n=32))
    RE(agent.learn(acqf, n=1))
    getattr(agent, acqf)
