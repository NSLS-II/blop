import pytest


@pytest.mark.test_func
def test_pareto(agent_2d_2f, RE):
    agent = agent_2d_2f
    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()


@pytest.mark.parametrize("acqf", ["qnehvi"])
def test_monte_carlo_pareto_acqfs(agent_2d_2f, RE, acqf):
    agent = agent_2d_2f
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=2))


@pytest.mark.test_func
def test_constrained_pareto(agent_2d_2f_2c, RE):
    agent = agent_2d_2f_2c
    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()


@pytest.mark.parametrize("acqf", ["qnehvi"])
def test_constrained_monte_carlo_pareto_acqfs(agent_2d_2f_2c, RE, acqf):
    agent = agent_2d_2f_2c
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=2))
