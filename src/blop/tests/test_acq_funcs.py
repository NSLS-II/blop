import pytest


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_one_dimensional(agent_1dof_1fit, RE, acqf):
    RE(agent_1dof_1fit.learn("qr", n=16))
    RE(agent_1dof_1fit.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_one_dimensional(agent_1dof_1fit, RE, acqf):
    RE(agent_1dof_1fit.learn("qr", n=16))
    RE(agent_1dof_1fit.learn(acqf, n=2))


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_single_objective(agent_2dof_1fit, RE, acqf):
    RE(agent_2dof_1fit.learn("qr", n=16))
    RE(agent_2dof_1fit.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_single_objective(agent_2dof_1fit, RE, acqf):
    RE(agent_2dof_1fit.learn("qr", n=16))
    RE(agent_2dof_1fit.learn(acqf, n=2))


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_multi_objective(agent_2dof_2fit, RE, acqf):
    RE(agent_2dof_2fit.learn("qr", n=16))
    RE(agent_2dof_2fit.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_multi_objective(agent_2dof_2fit, RE, acqf):
    RE(agent_2dof_2fit.learn("qr", n=16))
    RE(agent_2dof_2fit.learn(acqf, n=2))
