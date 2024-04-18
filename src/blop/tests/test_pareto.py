import pytest


@pytest.mark.test_func
def test_pareto(mo_agent, RE):
    RE(mo_agent.learn("qr", n=16))
    mo_agent.plot_pareto_front()


@pytest.mark.parametrize("acq_func", ["qnehvi"])
def test_monte_carlo_pareto_acq_funcs(mo_agent, RE, acq_func):
    RE(mo_agent.learn("qr", n=16))
    RE(mo_agent.learn(acq_func, n=2))


@pytest.mark.test_func
def test_constrained_pareto(constrained_agent, RE):
    RE(constrained_agent.learn("qr", n=16))
    constrained_agent.plot_pareto_front()


@pytest.mark.parametrize("acq_func", ["qnehvi"])
def test_constrained_monte_carlo_pareto_acq_funcs(constrained_agent, RE, acq_func):
    RE(constrained_agent.learn("qr", n=16))
    RE(constrained_agent.learn(acq_func, n=2))
