import pytest


@pytest.mark.test_func
def test_passive_dofs(agent_with_passive_dofs, RE):
    RE(agent_with_passive_dofs.learn("qr", n=32))

    agent_with_passive_dofs.plot_objectives()
    agent_with_passive_dofs.plot_acquisition()
    agent_with_passive_dofs.plot_constraint()
