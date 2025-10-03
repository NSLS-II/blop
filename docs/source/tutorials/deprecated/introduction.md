---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: blop
  language: python
  name: python3
---

# Introduction (Himmelblau's function)


+++

Let's use ``blop`` to minimize Himmelblau's function, which has four global minima:

+++

Select the type of data storage system you would like to use

```{code-cell} ipython3
from datetime import datetime

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer

from blop import DOF, Agent, Objective
from blop.sim import HDF5Handler
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer()
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


env = setup_re_env(db_type="temp", root_dir="/tmp/blop/sim", method="databroker")
globals().update(env)
bec.disable_plots()
```

```{code-cell} ipython3
import matplotlib as mpl

from blop.utils import functions

x1 = x2 = np.linspace(-6, 6, 256)
X1, X2 = np.meshgrid(x1, x2)

F = functions.himmelblau(X1, X2)

plt.pcolormesh(x1, x2, F, norm=mpl.colors.LogNorm(vmin=1e-1, vmax=1e3), cmap="magma_r")
plt.colorbar()
plt.xlabel("x1")
plt.ylabel("x2")
```

There are several things that our agent will need. The first ingredient is some degrees of freedom (these are always `ophyd` devices) which the agent will move around to different inputs within each DOF's bounds (the second ingredient). We define these here:

```{code-cell} ipython3
dofs = [
    DOF(name="x1", search_domain=(-6, 6)),
    DOF(name="x2", search_domain=(-6, 6)),
]
```

We also need to give the agent something to do. We want our agent to look in the feedback for a variable called 'himmelblau', and try to minimize it.

```{code-cell} ipython3
objectives = [Objective(name="himmelblau", description="Himmeblau's function", target="min")]
```

In our digestion function, we define our objective as a deterministic function of the inputs:

```{code-cell} ipython3
def digestion(df):
    df["himmelblau"] = []
    for val_x1, val_x2 in zip(df.get("x1"), df.get("x2"), strict=False):
        df["himmelblau"].append(functions.himmelblau(val_x1, val_x2))
    return df
```

We then combine these ingredients into an agent, giving it an instance of ``databroker`` so that it can see the output of the plans it runs.

```{code-cell} ipython3
agent = Agent(
    dofs=dofs,
    objectives=objectives,
    digestion=digestion,
    db=db,
)
```

Without any data, we can't make any inferences about what the function looks like, and so we can't use any non-trivial acquisition functions. Let's start by quasi-randomly sampling the parameter space, and plotting our model of the function:

```{code-cell} ipython3
RE(agent.learn("quasi-random", n=36))
agent.plot_objectives()
```

To decide which points to sample, the agent needs an acquisition function. The available acquisition function are here:

```{code-cell} ipython3
agent.all_acqfs
```

Now we can start to learn intelligently. Using the shorthand acquisition functions shown above, we can see the output of a few different ones:

```{code-cell} ipython3
agent.plot_acquisition(acqf="qei")
```

To decide where to go, the agent will find the inputs that maximize a given acquisition function:

```{code-cell} ipython3
agent.ask("qei", n=1)
```

We can also ask the agent for multiple points to sample and it will jointly maximize the acquisition function over all sets of inputs, and find the most efficient route between them:

```{code-cell} ipython3
res = agent.ask("qei", n=8, route=True)
agent.plot_acquisition(acqf="qei")
plt.scatter(res["points"]["x1"], res["points"]["x2"], marker="d", facecolor="w", edgecolor="k")
plt.plot(res["points"]["x1"], res["points"]["x2"], color="r")
```

All of this is automated inside the ``learn`` method, which will find a point (or points) to sample, sample them, and retrain the model and its hyperparameters with the new data. To do 4 learning iterations of 8 points each, we can run

```{code-cell} ipython3
RE(agent.learn("qei", n=4, iterations=8))
```

Our agent has found all the global minima of Himmelblau's function using Bayesian optimization, and we can ask it for the best point: 

```{code-cell} ipython3
agent.plot_objectives()
print(agent.best)
```
