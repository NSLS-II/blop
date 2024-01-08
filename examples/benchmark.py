"""
This script should be run inside of the IPython environment with pre-defined
objects RE, db, etc.

% run -i benchmark.py
"""

import time as ttime

import numpy as np

from blop import gp

bo = gp.BayesianOptimizer(
    init_scheme="quasi-random",
    n_init=64,
    detectors=[vstream, I0],
    shutter=psh,
    run_engine=RE,
    db=db,
    dofs=dofs,
    dof_bounds=hard_bounds,
    verbose=True,
)

timeout = 300
start_time = ttime.monotonic()
while ttime.monotonic() - start_time < timeout:
    bo.learn(
        n_iter=1,
        n_per_iter=16,
        strategy="ei",
        greedy=True,
    )

timestamps = bo.data.time.astype(int).values / 1e9

plt.plot(
    timestamps - timestamps[0],
    [
        np.nanmax(bo.data.fitness.values[: i + 1]) if not all(np.isnan(bo.data.fitness.values[: i + 1])) else np.nan
        for i in range(len(bo.data.fitness.values))
    ],
)

bo.data.drop(columns=[f"vstream_image"], inplace=True)
bo.data.to_hdf(f"/nsls2/data/tes/shared/config/gpo-benchmarks-230331/{int(timestamps[0])}.h5", "data")

del bo
