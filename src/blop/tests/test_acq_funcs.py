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
def test_analytic_acq_funcs_multi_objective(agent_with_multiple_objs, RE, acq_func):
    RE(agent_with_multiple_objs.learn("qr", n=16))
    RE(agent_with_multiple_objs.learn(acq_func, n=1))


@pytest.mark.parametrize("acq_func", ["qei", "qpi", "qem", "qucb"])
def test_monte_carlo_acq_funcs_multi_objective(agent_with_multiple_objs, RE, acq_func):
    RE(agent_with_multiple_objs.learn("qr", n=16))
    RE(agent_with_multiple_objs.learn(acq_func, n=4))
