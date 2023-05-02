import pytest

from bloptools.bo import BayesianOptimizationAgent
from bloptools.experiments.tests import himmelblau


@pytest.mark.test_func
def test_himmelblau_boa(RE, db):
    boa = BayesianOptimizationAgent(
        dofs=himmelblau.dofs,  # things which we move around
        bounds=himmelblau.bounds,  # how much we can move them
        dets=[],  # things to trigger
        tasks=[himmelblau.MinHimmelblau],  # tasks for the optimizer
        experiment=himmelblau,  # what experiment we're working on
        db=db,  # a databroker instance
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=8))
    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=3))

    boa.plot_tasks()
