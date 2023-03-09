import json
import time as ttime

import bluesky.plans as bp
import numpy as np
import pytest
from sirepo_bluesky.sirepo_ophyd import create_classes

from bloptools.gp import Optimizer


def test_shadow_gp_optimizer(RE, db, shadow_tes_simulation):
    data, schema = shadow_tes_simulation.auth("shadow", "00000002")
    classes, objects = create_classes(shadow_tes_simulation.data, connection=shadow_tes_simulation)
    globals().update(**objects)

    data["models"]["simulation"]["npoint"] = 100000
    data["models"]["watchpointReport12"]["histogramBins"] = 32

    dofs = [kbv.x_rot, kbv.offz]

    hard_bounds = np.array([[-0.10, +0.10], [-0.50, +0.50]])

    for dof in dofs:
        dof.kind = "hinted"

    gpo = Optimizer(
        init_scheme="quasi-random",
        n_init=4,
        run_engine=RE,
        db=db,
        detector=w9,
        detector_type="image",
        dofs=dofs,
        dof_bounds=hard_bounds,
        fitness_model="max_sep_density",
        training_iter=100,
        verbose=True,
    )

    gpo.learn(n_iter=1, n_per_iter=1, strategy="explore", greedy=True, reuse_hypers=False)
    gpo.learn(n_iter=1, n_per_iter=1, strategy="exploit", greedy=True, reuse_hypers=False)

    gpo.plot_state(gridded=True)
    gpo.plot_fitness()
