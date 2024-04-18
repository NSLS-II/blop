import pytest


@pytest.mark.test_func
def test_passive_dofs(agent_with_passive_dofs, RE):
    agent = agent_with_passive_dofs
    RE(agent.learn("qr", n=32))
    RE(agent.learn("qei", n=2))
