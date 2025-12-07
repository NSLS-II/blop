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

from blop.ax import Agent, RangeDOF, Objective

from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.callbacks.best_effort import BestEffortCallback
from tiled.client import from_uri
from tiled.client.container import Container
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
    RangeDOF(movable=x1, bounds=(-5, 5), parameter_type="float"),
    RangeDOF(movable=x2, bounds=(-5, 5), parameter_type="float"),
]
objectives = [
    Objective(name="himmelblau_2d", minimize=True),
]
readables = []
```

```{note}
Additional readables are typically added as a list of devices that produce data, such as detectors, to help with computing the desired outcome via the evaluation function.
```

Next, we will define the evaluation function. This is initialized with a `tiled_client`. Notice the care we take in handling
the evaluation of individual suggestions based on the `"_id"` key. This is important to consider when acquiring data for multiple suggestions in the same Bluesky run.

```{code-cell} ipython3
class Himmelblau2DEvaluation():
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        for suggestion in suggestions:
            # Special key to identify a suggestion
            suggestion_id = suggestion["_id"]
            x1 = x1_data[suggestion_id % len(x1_data)]
            x2 = x2_data[suggestion_id % len(x2_data)]
            outcomes.append({"himmelblau_2d": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2, "_id": suggestion_id})
        
        return outcomes
```

Next, we will setup the agent and perform the optimization using the run engine.
    
```{code-cell} ipython3
agent = Agent(
    readables=readables,
    dofs=dofs,
    objectives=objectives,
    evaluation=Himmelblau2DEvaluation(
        tiled_client=tiled_client,
    ),
    name="simple-experiment",
    description="A simple experiment optimizing the Himmelblau function",
)

RE(agent.optimize(30))
```

Now we can view the results.

```{code-cell} ipython3
agent.plot_objective("x1", "x2", "himmelblau_2d")
agent.ax_client.summarize()
```
