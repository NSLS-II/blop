import pytest

from bloptools import test_functions
from bloptools.bayesian import DOF, Agent, BrownianMotion, Objective


@pytest.mark.test_func
def test_passive_dofs(RE, db):
    dofs = [
        DOF(name="x1", limits=(-5.0, 5.0)),
        DOF(name="x2", limits=(-5.0, 5.0)),
        DOF(BrownianMotion(name="brownian1"), read_only=True),
        DOF(BrownianMotion(name="brownian1"), read_only=True),
    ]

    objectives = [
        Objective(key="himmelblau", minimize=True),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=test_functions.constrained_himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    RE(agent.learn("qr", n=32))

    agent.plot_tasks()
    agent.plot_acquisition()
    agent.plot_validity()
