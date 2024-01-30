import pytest  # noqa F401


def test_agent(agent, RE):
    RE(agent.learn("qr", n=4))


def test_forget(agent, RE):
    RE(agent.learn("qr", n=4))
    agent.forget(last=2)
