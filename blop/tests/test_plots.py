import pytest


@pytest.mark.test_func
def test_plots(RE, agent):
    RE(agent.learn("qr", n=32))

    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_constraint()
    agent.plot_history()
