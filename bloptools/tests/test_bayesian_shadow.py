import pytest
from sirepo_bluesky.sirepo_ophyd import create_classes

import bloptools
from bloptools.experiments.sirepo.tes import w9_digestion


@pytest.mark.shadow
def test_bayesian_agent_tes_shadow(RE, db, shadow_tes_simulation):
    data, schema = shadow_tes_simulation.auth("shadow", "00000002")
    classes, objects = create_classes(shadow_tes_simulation.data, connection=shadow_tes_simulation)
    globals().update(**objects)

    data["models"]["simulation"]["npoint"] = 100000
    data["models"]["watchpointReport12"]["histogramBins"] = 32

    dofs = [
        {"device": kbv.x_rot, "limits": (-0.1, 0.1), "kind": "active"},
        {"device": kbv.offz, "limits": (-0.5, 0.5), "kind": "active"},
    ]

    tasks = [
        {"key": "flux", "kind": "maximize"},
        {"key": "w9_fwhm_x", "kind": "minimize"},
        {"key": "w9_fwhm_y", "kind": "minimize"},
    ]

    agent = bloptools.bayesian.Agent(
        dofs=dofs,
        tasks=tasks,
        dets=[w9],
        digestion=w9_digestion,
        db=db,
    )

    RE(agent.initialize("qr", n_init=4))

    RE(agent.learn("ei", n_iter=2))
    RE(agent.learn("pi", n_iter=2))

    agent.plot_tasks()
