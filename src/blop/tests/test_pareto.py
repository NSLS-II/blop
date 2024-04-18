import pytest


@pytest.mark.test_func
def test_pareto(agent_2dof_2fit, RE):
    agent = agent_2dof_2fit
    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()

@pytest.mark.parametrize("acqf", ["qnehvi"])
def test_monte_carlo_pareto_acqfs(agent_2dof_2fit, RE, acqf):
    agent = agent_2dof_2fit
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=2))

@pytest.mark.test_func
def test_constrained_pareto(agent_2dof_2fit_2con, RE):
    agent = agent_2dof_2fit_2con
    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()

@pytest.mark.parametrize("acqf", ["qnehvi"])
def test_constrained_monte_carlo_pareto_acqfs(agent_2dof_2fit_2con, RE, acqf):
    agent = agent_2dof_2fit_2con
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=2))
