import numpy as np
import pytest
from sirepo_bluesky.sirepo_ophyd import create_classes

from bloptools.bayesian import Agent
from bloptools.experiments.sirepo import tes
from bloptools.tasks import Task


@pytest.mark.shadow
def test_bayesian_agent_tes_shadow(RE, db, shadow_tes_simulation):
    data, schema = shadow_tes_simulation.auth("shadow", "00000002")
    classes, objects = create_classes(shadow_tes_simulation.data, connection=shadow_tes_simulation)
    globals().update(**objects)

    data["models"]["simulation"]["npoint"] = 100000
    data["models"]["watchpointReport12"]["histogramBins"] = 32

    kb_dofs = [kbv.x_rot, kbv.offz]
    kb_bounds = np.array([[-0.10, +0.10], [-0.50, +0.50]])

    for dof in kb_dofs:
        dof.kind = "hinted"

    beam_flux_task = Task(key="flux", kind="max", transform=lambda x: np.log(x))
    beam_width_task = Task(key="x_width", kind="min", transform=lambda x: np.log(x))
    beam_height_task = Task(key="y_width", kind="min", transform=lambda x: np.log(x))

    boa = Agent(
        dofs=kb_dofs,
        bounds=kb_bounds,
        detectors=[w9],
        tasks=[beam_flux_task, beam_width_task, beam_height_task],
        digestion=tes.digestion,
        db=db,
    )

    RE(boa.initialize(init_scheme="quasi-random", n_init=4))

    RE(boa.learn(strategy="esti", n_iter=2, n_per_iter=2))

    boa.plot_tasks()
