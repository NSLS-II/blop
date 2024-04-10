import pytest


@pytest.mark.test_func
def test_pareto(mo_agent, RE):
    RE(mo_agent.learn("qr", n=16))

    mo_agent.plot_pareto_front()


@pytest.mark.parametrize("acq_func", ["qnehvi"])
def test_analytic_pareto_acq_funcs(mo_agent, RE, acq_func):
    RE(mo_agent.learn("qr", n=4))
    RE(mo_agent.learn(acq_func, n=2))
