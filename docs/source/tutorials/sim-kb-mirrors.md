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

Here we prepare the `RunEngine`, setup a local [Tiled](https://blueskyproject.io/tiled) server, and connect to it with a Tiled client.

```{code-cell} ipython3
import logging

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import matplotlib.pyplot as plt
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.sim.beamline import TiledBeamline

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)

DETECTOR_STORAGE = "/tmp/blop/sim"
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
bec = best_effort.BestEffortCallback()
bec.disable_plots()

RE = RunEngine({})
RE.subscribe(bec)
RE.subscribe(tiled_writer)
```

## Simulated beamline with KB mirror pair

Here we describe an analytical simulated beamline with a [KB mirror](https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Baez_mirror) pair. This is implemented as an [Ophyd](https://blueskyproject.io/ophyd/) device for ease-of-use with Bluesky.

```{code-cell} ipython3
beamline = TiledBeamline(name="bl")
```

## Create a Blop-Ax experiment

Now we can define the experiment we plan to run.

This involves setting 4 parameters that simulate motor positions controlling two KB mirrors. The objectives of the experiment are to maximize the beam intensity while minimizing the area of the beam.

We transform the Agent into an optimization problem that can be used with standard Bluesky plans.

```{code-cell} ipython3
from blop.ax import Agent, RangeDOF, Objective
from blop.protocols import EvaluationFunction

dofs = [
    RangeDOF(movable=beamline.kbv_dsv, parameter_type="float", bounds=(-5.0, 5.0)),
    RangeDOF(movable=beamline.kbv_usv, parameter_type="float", bounds=(-5.0, 5.0)),
    RangeDOF(movable=beamline.kbh_dsh, parameter_type="float", bounds=(-5.0, 5.0)),
    RangeDOF(movable=beamline.kbh_ush, parameter_type="float", bounds=(-5.0, 5.0)),
]

objectives = [
    Objective(name="bl_det_sum", minimize=False),
    Objective(name="bl_det_wid_x", minimize=True),
    Objective(name="bl_det_wid_y", minimize=True),
]

class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run = self.tiled_client[uid]
        bl_det_sum = run["primary/bl_det_sum"].read()
        bl_det_wid_x = run["primary/bl_det_wid_x"].read()
        bl_det_wid_y = run["primary/bl_det_wid_y"].read()

        # These ids are stored in the start document's metadata when
        # using the `blop.plans.default_acquire` plan.
        # You may want to store them differently in your experiment when writing
        # your a custom acquisiton plan.
        suggestion_ids = run.metadata["start"]["blop_suggestion_ids"]

        for idx, sid in enumerate(suggestion_ids):
            outcome = {
                "_id": sid,
                "bl_det_sum": bl_det_sum[idx],
                "bl_det_wid_x": bl_det_wid_x[idx],
                "bl_det_wid_y": bl_det_wid_y[idx],
            }
            outcomes.append(outcome)
        return outcomes

evaluation_function = DetectorEvaluation(tiled_client)

agent = Agent(
    readables=[beamline.det],
    dofs=dofs,
    objectives=objectives,
    evaluation=evaluation_function,
    name="sim_kb_mirror",
    description="Simulated KB Mirror Experiment",
)
```

## Optimization

With all of our experimental setup done, we can optimize the DOFs to satisfy our objectives.

For this example, Ax will optimize the 4 motor positions to produce the greatest intensity beam with the smallest beam width and height (smallest area). It does this by first running a couple of trials which are random samples, then the remainder using Bayesian optimization through BoTorch.

```{code-cell} ipython3
RE(agent.optimize(iterations=25, n_points=1))
```

## Analyze Results

We can start by summarizing each step of the optimization procedure and whether trials were successful or not. This can be done by accessing the Ax client directly.

```{code-cell} ipython3
agent.ax_client.summarize()
```

### Plotting

We also can plot slices of the parameter space with respect to our objectives.

```{code-cell} ipython3
from ax.analysis import SlicePlot

_ = agent.ax_client.compute_analyses(analyses=[SlicePlot("bl_kbv_dsv", "bl_det_sum")])
```

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses(analyses=[SlicePlot("bl_kbv_dsv", "bl_det_wid_x")])
```

### More comprehensive analyses

Ax provides many analysis tools that can help understand optimization results.

```{code-cell} ipython3
from ax.analysis import TopSurfacesAnalysis

_ = agent.ax_client.compute_analyses(analyses=[TopSurfacesAnalysis("bl_det_sum")])
```

### Visualizing the optimal beam

Below we get the optimal parameters, move the motors to their optimal positions, and observe the resulting beam.

```{code-cell} ipython3
optimal_parameters = next(iter(agent.ax_client.get_pareto_frontier()))[0]
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
image = tiled_client[uid[0]]["primary/bl_det_image"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.show()
```
