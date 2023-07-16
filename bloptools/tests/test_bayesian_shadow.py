import numpy as np
import pytest
from sirepo_bluesky.sirepo_ophyd import create_classes

import bloptools
from bloptools.experiments.sirepo.tes import w9_digestion
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

    beam_flux_task = Task(key="flux", kind="max", transform=lambda x: np.log(x))
    beam_width_task = Task(key="x_width", kind="min", transform=lambda x: np.log(x))
    beam_height_task = Task(key="y_width", kind="min", transform=lambda x: np.log(x))

    agent = bloptools.bayesian.Agent(
        active_dofs=kb_dofs,
        passive_dofs=[],
        detectors=[w9],
        active_dof_bounds=kb_bounds,
        tasks=[beam_flux_task, beam_width_task, beam_height_task],
        digestion=w9_digestion,
        db=db,
    )

    RE(agent.initialize(acqf="qr", n_init=4))

    RE(agent.learn(acqf="ei", n_iter=2))
    RE(agent.learn(acqf="pi", n_iter=2))

    agent.plot_tasks()
