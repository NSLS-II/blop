import pytest


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_one_dimensional(agent_1d_1f, RE, acqf):
    RE(agent_1d_1f.learn("qr", n=16))
    RE(agent_1d_1f.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_one_dimensional(agent_1d_1f, RE, acqf):
    RE(agent_1d_1f.learn("qr", n=16))
    RE(agent_1d_1f.learn(acqf, n=2))


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_single_objective(agent_2d_1f, RE, acqf):
    RE(agent_2d_1f.learn("qr", n=16))
    RE(agent_2d_1f.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_single_objective(agent_2d_1f, RE, acqf):
    RE(agent_2d_1f.learn("qr", n=16))
    RE(agent_2d_1f.learn(acqf, n=2))


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_multi_objective(agent_2d_2f, RE, acqf):
    RE(agent_2d_2f.learn("qr", n=16))
    RE(agent_2d_2f.learn(acqf, n=1))


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_multi_objective(agent_2d_2f, RE, acqf):
    RE(agent_2d_2f.learn("qr", n=16))
    RE(agent_2d_2f.learn(acqf, n=2))
