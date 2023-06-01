import pytest

import bloptools
from bloptools.experiments.tests import himmelblau_digestion, mock_kbs_digestion
from bloptools.tasks import Task


@pytest.mark.test_func
def test_bayesian_agent_himmelblau(RE, db):
    dofs = bloptools.experiments.tests.get_dummy_dofs(n=2)  # get a list of two DOFs
    bounds = [(-5.0, +5.0), (-5.0, +5.0)]
    task = Task(key="himmelblau", kind="min")

    boa = bloptools.bayesian.Agent(
        dofs=dofs,
        bounds=bounds,
        tasks=task,
        digestion=himmelblau_digestion,
        db=db,
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=16))

    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=3))

    boa.plot_tasks()


@pytest.mark.test_func
def test_bayesian_agent_mock_kbs(RE, db):
    dofs = bloptools.experiments.tests.get_dummy_dofs(n=4)  # get a list of two DOFs
    bounds = [(-4.0, +4.0), (-4.0, +4.0), (-4.0, +4.0), (-4.0, +4.0)]

    tasks = [Task(key="x_width", kind="min"), Task(key="y_width", kind="min")]

    boa = bloptools.bayesian.Agent(
        dofs=dofs,
        bounds=bounds,
        tasks=tasks,
        digestion=mock_kbs_digestion,
        db=db,
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=16))

    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=3))

    boa.plot_tasks()
