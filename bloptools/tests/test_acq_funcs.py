import pytest


@pytest.mark.test_func
def test_acq_funcs_single_task(agent, RE, db):
    RE(agent.learn("qr", n=32))

    # analytic methods
    RE(agent.learn("ei", n=1))
    RE(agent.learn("pi", n=1))
    RE(agent.learn("em", n=1))
    RE(agent.learn("ucb", n=1))

    RE(agent.learn("qei", n=2))
    RE(agent.learn("qpi", n=2))
    RE(agent.learn("qem", n=2))
    RE(agent.learn("qucb", n=2))
