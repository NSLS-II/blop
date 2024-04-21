import pytest  # noqa F401


def test_plots(RE, agent):
    RE(agent.learn("qr", n=16))

    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()


def test_plots_multiple_objs(RE, mo_agent):
    RE(mo_agent.learn("qr", n=16))

    mo_agent.plot_objectives()
    mo_agent.plot_acquisition()
    mo_agent.plot_validity()
    mo_agent.plot_history()


def test_plots_read_only_dofs(RE, agent_with_read_only_dofs):
    RE(agent_with_read_only_dofs.learn("qr", n=16))

    agent_with_read_only_dofs.plot_objectives()
    agent_with_read_only_dofs.plot_acquisition()
    agent_with_read_only_dofs.plot_validity()
    agent_with_read_only_dofs.plot_history()
