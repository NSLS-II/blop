---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Simulated KB Mirror Demo

This notebook introduces the use of Blop to tune a KB mirror pair to optimize the quality of a simulated beam read by a detector.

Blop uses [Ax](https://ax.dev) as its optimization and experiment tracking backend.

Ax provides:
- Experiment tracking
- Analysis & visualization
- Bayesian optimization (through [BoTorch](https://botorch.org/))

Blop provides:
- Native integration with [Bluesky & its ecosystem](https://blueskyproject.io)
- Specialized kernels and methods common to beamline optimization problems

These features make it simple to optimize your beamline using both Bluesky & Ax.

+++

## Preparing a test environment

Here we prepare the `RunEngine`.

```{code-cell} ipython3
from datetime import datetime

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.sim import HDF5Handler
from blop.sim.beamline import DatabrokerBeamline, TiledBeamline

DETECTOR_STORAGE = "/tmp/blop/sim"
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)


def setup_re_env(db_type="default", root_dir="/default/path", method="tiled"):
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)
    _ = make_dir_tree(datetime.now().year, base_path=root_dir)
    if method.lower() == "tiled":
        RE.subscribe(tiled_writer)
        return {"RE": RE, "db": tiled_client, "bec": bec}
    elif method.lower() == "databroker":
        db = Broker.named(db_type)
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        RE.subscribe(db.insert)
        return {"RE": RE, "db": db, "bec": bec, }
    else:
        raise ValueError("The method for data storage used is not supported")


def register_handlers(db, handlers):
    for handler_spec, handler_class in handlers.items():
        db.reg.register_handler(handler_spec, handler_class, overwrite=True)


env = setup_re_env(db_type="temp", root_dir="/tmp/blop/sim", method="tiled")
globals().update(env)
bec.disable_plots()
```

## Simulated beamline with KB mirror pair

Here we describe an analytical simulated beamline with a [KB mirror](https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Baez_mirror) pair. This is implemented as an [Ophyd](https://blueskyproject.io/ophyd/) device for ease-of-use with Bluesky.

```{code-cell} ipython3
if isinstance(db, Container):
    beamline = TiledBeamline(name="bl")
elif isinstance(db, databroker.v1.Broker):
    beamline = DatabrokerBeamline(name="bl")
```

## Create a Blop-Ax experiment

Now we can define the experiment we plan to run.

This involves setting 4 parameters that simulate motor positions controlling two KB mirrors. The objectives of the experiment are to maximize the beam intensity while minimizing the area of the beam.

```{code-cell} ipython3
from blop.ax import Agent
from blop.dofs import DOF
from blop.objectives import Objective

dofs = [
    DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
    DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
    DOF(movable=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
    DOF(movable=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
]

objectives = [
    Objective(name="bl_det_sum", target="max"),
    Objective(name="bl_det_wid_x", target="min"),
    Objective(name="bl_det_wid_y", target="min"),
]

agent = Agent(
    readables=[beamline.det],
    dofs=dofs,
    objectives=objectives,
    db=db,
)
agent.configure_experiment(name="test_ax_agent", description="Test the AxAgent")
```

## Optimization

With all of our experimental setup done, we can optimize the DOFs to satisfy our objectives.

For this example, Ax will optimize the 4 motor positions to produce the greatest intensity beam with the smallest beam width and height (smallest area). It does this by first running a couple of trials which are random samples, then the remainder using Bayesian optimization through BoTorch.

```{code-cell} ipython3
RE(agent.learn(iterations=25, n=1))
```

## Analyze Results

We can start by summarizing each step of the optimization procedure and whether trials were successful or not.

```{code-cell} ipython3
agent.summarize()
```

### Plotting

We also can plot slices of the parameter space with respect to our objectives.

```{code-cell} ipython3
from ax.analysis import SlicePlot

_ = agent.compute_analyses(analyses=[SlicePlot("bl_kbv_dsv", "bl_det_sum")])
```

```{code-cell} ipython3
_ = agent.compute_analyses(analyses=[SlicePlot("bl_kbv_dsv", "bl_det_wid_x")])
```

### More comprehensive analyses

Ax provides many analysis tools that can help understand optimization results.

```{code-cell} ipython3
from ax.analysis import TopSurfacesAnalysis

_ = agent.compute_analyses(analyses=[TopSurfacesAnalysis("bl_det_sum")])
```

### Visualizing the optimal beam

Below we get the optimal parameters, move the motors to their optimal positions, and observe the resulting beam.

```{code-cell} ipython3
optimal_parameters = next(iter(agent.client.get_pareto_frontier()))[0]
optimal_parameters
```

```{code-cell} ipython3
from bluesky.plans import list_scan

scan_motor_params = []
for motor in [beamline.kbv_dsv, beamline.kbv_usv, beamline.kbh_dsh, beamline.kbh_ush]:
    scan_motor_params.append(motor)
    scan_motor_params.append([optimal_parameters[motor.name]])
uid = RE(list_scan([beamline.det], *scan_motor_params))
```

```{code-cell} ipython3

image = db[uid[0]]["primary"]["bl_det_image"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.show()
```
