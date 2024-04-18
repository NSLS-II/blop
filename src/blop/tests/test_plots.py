import pytest  # noqa F401


def test_plots_one_dimensional(RE, agent_1dof_1fit):
    agent = agent_1dof_1fit
    RE(agent.learn("qr", n=16))
    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()

def test_plots_two_dimensional(RE, agent_1dof_1fit):
    agent = agent_1dof_1fit
    RE(agent.learn("qr", n=16))
    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()


def test_plots_multiple_objs(RE, agent_2dof_2fit):
    agent = agent_2dof_2fit
    RE(agent.learn("qr", n=16))
    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()


def test_plots_read_only_dofs(RE, agent_with_read_only_dofs):
    RE(agent_with_read_only_dofs.learn("qr", n=16))

    agent_with_read_only_dofs.plot_objectives()
    agent_with_read_only_dofs.plot_acquisition()
    agent_with_read_only_dofs.plot_validity()
    agent_with_read_only_dofs.plot_history()
