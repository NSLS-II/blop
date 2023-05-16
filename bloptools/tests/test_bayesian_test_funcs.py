import pytest

import bloptools
from bloptools.experiments.tests import himmelblau, mock_kbs


@pytest.mark.test_func
def test_bayesian_agent_himmelblau(RE, db):
    boa = bloptools.bayesian.Agent(
        dofs=himmelblau.dofs,  # things which we move around
        bounds=himmelblau.bounds,  # how much we can move them
        tasks=[himmelblau.MinHimmelblau],  # tasks for the optimizer
        acquisition=himmelblau.acquisition,  # what experiment we're working on
        digestion=himmelblau.digestion,
        db=db,  # a databroker instance
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=16))

    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=3))

    boa.plot_tasks()


@pytest.mark.test_func
def test_bayesian_agent_mock_kbs(RE, db):
    boa = bloptools.bayesian.Agent(
        dofs=mock_kbs.dofs,  # things which we move around
        bounds=mock_kbs.bounds,  # how much we can move them
        tasks=[mock_kbs.MinBeamWidth, mock_kbs.MinBeamHeight],  # tasks for the optimizer
        acquisition=mock_kbs.acquisition,  # what experiment we're working on
        digestion=mock_kbs.digestion,
        db=db,  # a databroker instance
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=16))

    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=3))

    boa.plot_tasks()
