import pytest  # noqa F401


def test_agent_2d_1f_save_load_data(agent_2d_1f, RE):
    RE(agent_2d_1f.learn("qr", n=4))
    agent_2d_1f.save_data("/tmp/test_save_data.h5")
    agent_2d_1f.reset()
    agent_2d_1f.load_data("/tmp/test_save_data.h5")
    RE(agent_2d_1f.learn("qr", n=4))


def test_agent_2d_1f_save_load_hypers(agent_2d_1f, RE):
    RE(agent_2d_1f.learn("qr", n=4))
    agent_2d_1f.save_hypers("/tmp/test_save_hypers.h5")
    agent_2d_1f.reset()
    RE(agent_2d_1f.learn("qr", n=16, hypers="/tmp/test_save_hypers.h5"))
