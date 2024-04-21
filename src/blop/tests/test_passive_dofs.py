import pytest


@pytest.mark.test_func
def test_read_only_dofs(agent_with_read_only_dofs, RE):
    RE(agent_with_read_only_dofs.learn("qr", n=32))
    RE(agent_with_read_only_dofs.learn("qei", n=2))
