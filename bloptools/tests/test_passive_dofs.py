import pytest

from bloptools import devices, test_functions
from bloptools.bayesian import Agent


@pytest.mark.test_func
def test_passive_dofs(RE, db):
    dofs = [
        {"device": devices.DOF(name="x1"), "limits": (-5, 5), "kind": "active"},
        {"device": devices.DOF(name="x2"), "limits": (-5, 5), "kind": "active"},
        {"device": devices.BrownianMotion(name="brownian1"), "limits": (-2, 2), "kind": "passive"},
        {"device": devices.BrownianMotion(name="brownian2"), "limits": (-2, 2), "kind": "passive"},
    ]

    tasks = [
        {"key": "himmelblau", "kind": "minimize"},
    ]

    agent = Agent(
        dofs=dofs,
        tasks=tasks,
        digestion=test_functions.himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    RE(agent.initialize("qr", n_init=32))

    agent.plot_tasks()
    agent.plot_acquisition()
    agent.plot_feasibility()
