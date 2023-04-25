import pytest

from bloptools.bo import BayesianOptimizationAgent
from bloptools.experiments.tests import Himmelblau


@pytest.mark.test_func
def test_himmelblau_boa(RE, db):
    himmelblau = Himmelblau()

    boa = BayesianOptimizationAgent(
        dofs=himmelblau.dofs,
        dets=[],
        bounds=himmelblau.bounds,
        db=db,
        experiment=himmelblau,
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=8))

    RE(boa.learn(strategy="eI", n_iter=2, n_per_iter=3))
    RE(boa.learn(strategy="eGIBBON", n_iter=2, n_per_iter=3))

    boa.plot_state(gridded=True)
