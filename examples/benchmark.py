"""
This script should be run inside of the IPython environment with pre-defined
objects RE, db, etc.

% run -i benchmark.py
"""

import time as ttime

import numpy as np

from bloptools import gp

gpo = gp.Optimizer(
    init_scheme="quasi-random",
    n_init=16,
    run_engine=RE,
    db=db,
    shutter=psh,
    detector=vstream,
    detector_type="image",
    dofs=dofs,
    dof_bounds=hard_bounds,
    fitness_model="max_sep_density",
    training_iter=512,
    verbose=True,
)

timeout = 300
start_time = ttime.monotonic()
while ttime.monotonic() - start_time < timeout:
    gpo.learn(
        n_iter=1, n_per_iter=5, strategy="explore", greedy=True, reuse_hypers=False, plots=["fitness"], upsample=2
    )
    gpo.learn(
        n_iter=1, n_per_iter=5, strategy="exploit", greedy=True, reuse_hypers=False, plots=["fitness"], upsample=2
    )

timestamps = gpo.data.time.astype(int).values / 1e9

plt.plot(timestamps[1:] - timestamps[0], [np.nanmax(gpo.fitness[:i]) for i in range(1, len(gpo.fitness))])

gpo.data["fitness"] = gpo.fitness
gpo.data.drop(columns=[f"{gpo.detector.name}_image"], inplace=True)
gpo.data.to_hdf(f"/nsls2/data/tes/shared/config/gpo-benchmarks-230218/{int(timestamps[0])}.h5", "data")

del gpo
