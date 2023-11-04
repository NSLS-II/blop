import pytest

from bloptools.bayesian import DOF, Agent, BrownianMotion, Objective
from bloptools.utils import functions


@pytest.mark.test_func
def test_passive_dofs(RE, db):
    dofs = [
        DOF(name="x1", limits=(-5.0, 5.0)),
        DOF(name="x2", limits=(-5.0, 5.0)),
        DOF(name="x3", limits=(-5.0, 5.0), active=False),
        DOF(BrownianMotion(name="brownian1"), read_only=True),
        DOF(BrownianMotion(name="brownian2"), read_only=True, active=False),
    ]

    objectives = [
        Objective(key="himmelblau", minimize=True),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=functions.constrained_himmelblau_digestion,
        db=db,
        verbose=True,
        tolerate_acquisition_errors=False,
    )

    RE(agent.learn("qr", n=32))

    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_constraint()
