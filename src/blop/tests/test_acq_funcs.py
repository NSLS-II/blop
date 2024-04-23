import pytest


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_one_dimensional(agent_1d_1f, RE, acqf):
    a = agent_1d_1f
    RE(a.learn("qr", n=16))
    RE(a.learn(acqf, n=1))
    getattr(a, acqf)


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_one_dimensional(agent_1d_1f, RE, acqf):
    a = agent_1d_1f
    RE(a.learn("qr", n=16))
    RE(a.learn(acqf, n=2))
    getattr(a, acqf)


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_single_objective(agent_2d_1f, RE, acqf):
    a = agent_2d_1f
    RE(a.learn("qr", n=4))
    RE(a.learn(acqf, n=1))
    getattr(a, acqf)


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_single_objective(agent_2d_1f, RE, acqf):
    a = agent_2d_1f
    RE(a.learn("qr", n=4))
    RE(a.learn(acqf, n=2))
    getattr(a, acqf)


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
def test_analytic_acqfs_multi_objective(agent_2d_2f, RE, acqf):
    a = agent_2d_2f
    RE(a.learn("qr", n=4))
    RE(a.learn(acqf, n=1))
    getattr(a, acqf)


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acqfs_multi_objective(agent_2d_2f, RE, acqf):
    a = agent_2d_2f
    RE(a.learn("qr", n=4))
    RE(a.learn(acqf, n=2))
    getattr(a, acqf)
