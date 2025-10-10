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

# Run a simple experiment with Bluesky

This guide will walk you through the process of running a simple experiment using Blop, Bluesky, and Tiled. See [https://blueskyproject.io/](https://blueskyproject.io/) for more information on the Bluesky ecosystem and how these tools work together.

Bluesky along with a data access backend (either Tiled or Databroker) are not necessary for using Blop, but are fully integrated into the package.

We'll start by importing the necessary libraries.

```{code-cell} ipython3
import logging
import time
from typing import Any

from blop import DOF, Objective
from blop.ax import Agent
from blop.dofs import DOF
from blop.objectives import Objective

from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.callbacks.best_effort import BestEffortCallback
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)
```

First, we will start up a Tiled server. This will act as our data access service which Bluesky will write to and that Blop can read from.

```{code-cell} ipython3
tiled_server = SimpleTiledServer()
```

Next, we'll set up the Bluesky run engine and connect to the local Tiled server.

```{code-cell} ipython3
RE = RunEngine({})
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
RE.subscribe(tiled_writer)
bec = BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)
```

In order to control parameters and acquire data with Bluesky, we must follow the `NamedMovable` and `Readable` protocols. To do this, we implement a simple class that implements both protocols. An alternative to implementing these protocols yourself is to use Ophyd. The additional `AlwaysSuccessfulStatus` is necessary to tell the Bluesky RunEngine when a move is complete. For the purposes of this tutorial, every move is successful and complete immediately.

```{code-cell} ipython3
class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout = 0.0):
        return None
    
    @property
    def done(self) -> bool:
        return True
    
    @property
    def success(self) -> bool:
        return True

class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return { 
            "fields": [self._name],
            "dimensions": [],
            "gridding": "rectilinear",
        }
    
    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {
            self._name: { "value": self._value, "timestamp": time.time() }
        }

    def describe(self):
        return {
            self._name: { "source": self._name, "dtype": "number", "shape": [] }
        }

class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()
```

    
Next, we'll define the DOFs and optimization objective. Since we can calculate our objective based on the two movable signals, there is no need to acquire data using an extra readable. With the movables already configured via the `DOF`s, it is implicitly added as a readable during the data acquisition.

```{code-cell} ipython3
x1 = MovableSignal("x1", initial_value=0.1)
x2 = MovableSignal("x2", initial_value=0.23)

dofs = [
    DOF(movable=x1, search_domain=(-5, 5)),
    DOF(movable=x2, search_domain=(-5, 5)),
]
objectives = [
    Objective(name="himmelblau_2d", target="min"),
]
readables = []
```

```{note}
Additional readables are typically added as a list of devices that produce data, such as detectors, to help with computing the desired outcome via the digestion function.
```

Next, we will define the digestion function. The data that will be available to the digestion function will always be a collection of readables (specified either implicitly or explicitly).

```{code-cell} ipython3
def himmelblau_2d_digestion(trial_index: int, data: dict[str, Any]) -> float:
    x1 = data["x1"][trial_index % len(data["x1"])]
    x2 = data["x2"][trial_index % len(data["x2"])]
    return {"himmelblau_2d": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2}
```

Next, we will setup the agent and perform the optimization using the run engine.
    
```{code-cell} ipython3
agent = Agent(readables=readables, dofs=dofs, objectives=objectives, db=tiled_client, digestion=himmelblau_2d_digestion)
agent.configure_experiment(name="simple_experiment", description="A simple experiment.")
RE(agent.learn(iterations=30))
```

Now we can view the results.

```{code-cell} ipython3
agent.plot_objective("x1", "x2", "himmelblau_2d")
agent.summarize()
```
