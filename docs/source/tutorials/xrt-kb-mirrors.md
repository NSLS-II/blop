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

# XRT KB Mirror Demo

+++

For ophyd beamline setup see: 
- https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_beamline.py
- https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_kb_model.py

The picture below displays beam from geometric source propagating through a pair of toroidal mirrors focusing the beam on screen. Simulation of a KB setup.

![xrt_blop_layout_w.jpg](../_static/xrt_blop_layout_w.jpg)

```{code-cell} ipython3
import time
from datetime import datetime

import plotly.io as pio

pio.renderers.default = "notebook"
import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import tiled.client.container
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer

from blop import DOF, Objective
from blop.ax import Agent
from blop.sim import HDF5Handler
from blop.sim.xrt_beamline import DatabrokerBeamline, TiledBeamline

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

    if method == "tiled":
        RE.subscribe(tiled_writer)
        return {"RE": RE, "db": tiled_client, "bec": bec}

    elif method == "databroker":
        db = Broker.named(db_type)
        db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
        try:
            databroker.assets.utils.install_sentinels(db.reg.config, version=1)
        except Exception:
            pass
        RE.subscribe(db.insert)
        return {
            "RE": RE,
            "db": db,
            "bec": bec,
        }
    else:
        raise ValueError("The method for data storage used is not supported")


def register_handlers(db, handlers):
    for handler_spec, handler_class in handlers.items():
        db.reg.register_handler(handler_spec, handler_class, overwrite=True)


env = setup_re_env(db_type="temp", root_dir="/tmp/blop/sim", method="tiled")
globals().update(env)
bec.disable_plots()
```

```{code-cell} ipython3
plt.ion()

h_opt = 0
dh = 5

R1, dR1 = 40000, 10000
R2, dR2 = 20000, 10000
```

```{code-cell} ipython3
if isinstance(db, tiled.client.container.Container):
    beamline = TiledBeamline(name="bl")
else:
    beamline = DatabrokerBeamline(name="bl")
time.sleep(1)
dofs = [
    DOF(description="KBV R", device=beamline.kbv_dsv, search_domain=(R1 - dR1, R1 + dR1)),
    DOF(description="KBH R", device=beamline.kbh_dsh, search_domain=(R2 - dR2, R2 + dR2)),
]
```

```{code-cell} ipython3
objectives = [
    Objective(name="bl_det_sum", target="max", transform="log", trust_domain=(20, 1e12)),
    Objective(
        name="bl_det_wid_x",
        target="min",
        transform="log",
        # trust_domain=(0, 1e12),
        # latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
    Objective(
        name="bl_det_wid_y",
        target="min",
        transform="log",
        # trust_domain=(0, 1e12),
        # latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
]
```

```{code-cell} ipython3
agent = Agent(
    readables=[beamline.det],
    dofs=dofs,
    objectives=objectives,
    db=db,
)
agent.configure_experiment(
    name="xrt-blop-demo",
    description="A demo of the Blop agent with XRT simulated beamline",
    experiment_type="demo",
)
```

```{code-cell} ipython3
RE(agent.learn(iterations=25))
```

```{code-cell} ipython3
_ = agent.plot_objective(x_dof_name="bl_kbh_dsh", y_dof_name="bl_kbv_dsv", objective_name="bl_det_sum")
```

## Visualizing the optimal beam

Below we get the optimal parameters, move the motors to their optimal positions, and observe the resulting beam.

```{code-cell} ipython3
optimal_parameters = next(iter(agent.client.get_pareto_frontier()))[0]
optimal_parameters
```

```{code-cell} ipython3
from bluesky.plans import list_scan

scan_motor_params = []
for motor in [beamline.kbv_dsv, beamline.kbh_dsh]:
    scan_motor_params.append(motor)
    scan_motor_params.append([optimal_parameters[motor.name]])
uid = RE(list_scan([beamline.det], *scan_motor_params))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

image = db[uid[0]]["streams"]["primary"]["bl_det_image"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.show()
```
