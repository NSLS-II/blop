import pytest


@pytest.mark.parametrize("acq_func", ["ei", "pi", "em", "ucb"])
def test_analytic_acq_funcs_single_objective(agent, RE, acq_func):
    RE(agent.learn("qr", n=4))
    RE(agent.learn(acq_func, n=1))


@pytest.mark.parametrize("acq_func", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acq_funcs_single_objective(agent, RE, acq_func):
    RE(agent.learn("qr", n=4))
    RE(agent.learn(acq_func, n=4))


@pytest.mark.parametrize("acq_func", ["ei", "pi", "em", "ucb"])
def test_analytic_acq_funcs_multi_objective(mo_agent, RE, acq_func):
    RE(mo_agent.learn("qr", n=16))
    RE(mo_agent.learn(acq_func, n=1))


@pytest.mark.parametrize("acq_func", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acq_funcs_multi_objective(mo_agent, RE, acq_func):
    RE(mo_agent.learn("qr", n=16))
    RE(mo_agent.learn(acq_func, n=4))
