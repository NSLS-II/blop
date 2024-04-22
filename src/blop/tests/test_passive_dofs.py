import pytest


@pytest.mark.test_func
def test_read_only_dofs(agent_with_read_only_dofs, RE):
    agent = agent_with_read_only_dofs
    RE(agent.learn("qr", n=32))
    RE(agent.learn("qei", n=2))
