import pytest

import bloptools
from bloptools.tasks import Task
from bloptools.test_functions import himmelblau_digestion, mock_kbs_digestion


@pytest.mark.test_func
def test_bayesian_agent_himmelblau(RE, db):
    dofs = bloptools.devices.dummy_dofs(n=2)  # get a list of two DOFs
    bounds = [(-5.0, +5.0), (-5.0, +5.0)]
    task = Task(key="himmelblau", kind="min")

    agent = bloptools.bayesian.Agent(
        active_dofs=dofs,
        passive_dofs=[],
        active_dof_bounds=bounds,
        tasks=[task],
        digestion=himmelblau_digestion,
        db=db,
    )

    RE(agent.initialize("qr", n_init=16))

    RE(agent.learn("ei", n_iter=2))

    agent.plot_tasks()


@pytest.mark.test_func
def test_bayesian_agent_mock_kbs(RE, db):
    dofs = bloptools.devices.dummy_dofs(n=4)  # get a list of two DOFs
    bounds = [(-4.0, +4.0), (-4.0, +4.0), (-4.0, +4.0), (-4.0, +4.0)]

    tasks = [Task(key="x_width", kind="min"), Task(key="y_width", kind="min")]

    agent = bloptools.bayesian.Agent(
        active_dofs=dofs,
        passive_dofs=[],
        active_dof_bounds=bounds,
        tasks=tasks,
        digestion=mock_kbs_digestion,
        db=db,
    )

    RE(agent.initialize("qr", n_init=16))

    RE(agent.learn("ei", n_iter=4))

    agent.plot_tasks()
