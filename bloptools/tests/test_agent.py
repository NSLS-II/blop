import pytest


@pytest.mark.agent
def test_writing_hypers(RE, agent):
    RE(agent.initialize("qr", n_init=32))

    agent.save_hypers("hypers.h5")

    RE(agent.initialize("qr", n_init=8, hypers="hypers.h5"))
