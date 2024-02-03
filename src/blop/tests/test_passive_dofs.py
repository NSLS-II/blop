import pytest


@pytest.mark.test_func
def test_passive_dofs(agent_with_passive_dofs, RE):
    RE(agent_with_passive_dofs.learn("qr", n=32))
