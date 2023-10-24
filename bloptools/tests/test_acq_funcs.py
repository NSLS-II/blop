import pytest


@pytest.mark.parametrize("acq_func", ["ei", "pi", "em", "ucb"])
def test_analytic_acq_funcs_single_task(agent, RE, acq_func):
    RE(agent.learn("qr", n=32))
    RE(agent.learn(acq_func, n=1))


@pytest.mark.parametrize("acq_func", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acq_funcs_single_task(agent, RE, acq_func):
    RE(agent.learn("qr", n=32))
    RE(agent.learn(acq_func, n=4))
