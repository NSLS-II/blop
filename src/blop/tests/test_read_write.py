import pytest  # noqa F401


def test_agent_save_load_data(agent, RE):
    RE(agent.learn("qr", n=4))
    agent.save_data("/tmp/test_save_data.h5")
    agent.reset()
    agent.load_data("/tmp/test_save_data.h5")
    RE(agent.learn("qr", n=4))


def test_agent_save_load_hypers(agent, RE):
    RE(agent.learn("qr", n=4))
    agent.save_hypers("/tmp/test_save_hypers.h5")
    agent.reset()
    RE(agent.learn("qr", n=16, hypers="/tmp/test_save_hypers.h5"))
