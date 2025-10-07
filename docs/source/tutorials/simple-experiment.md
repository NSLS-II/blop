# Run a simple experiment with Bluesky

This guide will walk you through the process of running a simple experiment using Blop, Bluesky, and Tiled. See https://blueskyproject.io/ for more information on the Bluesky ecosystem and how these tools work together.

Bluesky along with a data access backend (either Tiled or Databroker) are not necessary for using Blop, but are fully integrated into the package.

We'll start by importing the necessary libraries.

```{code-cell} ipython3
    from blop import DOF, Objective
    from blop.ax import Agent
    from blop.dofs import DOF
    from blop.objectives import Objective

    from bluesky.protocols import NamedMovable, Readable
    from bluesky.run_engine import RunEngine
    from bluesky.callbacks import TiledWriter
    from tiled.client import from_uri
```

Next, we'll set up the Bluesky run engine, and connect to a local Tiled server.

```{code-cell} ipython3

    RE = RunEngine({})
    tiled_client = from_uri("http://localhost:8000")
    tiled_writer = TiledWriter(tiled_client)
    RE.subscribe(tiled_writer)
```

In order to control parameters and acquire data with Bluesky, we must follow the `NamedMovable` and `Readable` protocols. To do this, we implement a simple class that implements both protocols. An alternative to implementing these protocols yourself is to use Ophyd. The additional `AlwaysSuccessfulStatus` is necessary to tell the Bluesky RunEngine when a move is complete. For the purposes of this tutorial, every move is successful and complete immediately.

```{code-cell} ipython3

class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        pass

    def exception(self, timeout = 0.0):
        pass
    
    @property
    def done(self) -> bool:
        return True
    
    @property
    def success(self) -> bool:
        return True

class ReadableSignal(Readable):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

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

    
Next, we'll define the DOFs and optimization objective. Since we can calculate our objective based on the two movable signals, there is no need to acquire data using an explicit readable. With the movables already configured via the `DOF`s, it is implicitly added as a readable (since it also implements the `Readable` protocol).

.. code-block:: python
    x1 = MovableSignal("x1", initial_value=0.1)
    x2 = MovableSignal("x2", initial_value=0.23)

    dofs = [
        DOF(movable=x1, search_domain=(0, 10)),
        DOF(movable=x2, search_domain=(5, 15)),
    ]
    objectives = [
        Objective(name="himmelblau_2d", target="min"),
    ]
    readables = []

.. note:: Additional readables are typically added as a list of devices that produce data, such as detectors, to help with computing the desired outcome via the digestion function.

And finally, we'll create the agent, configure the experiment, and run it.
    
.. code-block:: python

    agent = Agent(readables=readables, dofs=dofs, objectives=objectives, db=tiled_client)
    agent.configure_experiment(name="simple_experiment", description="A simple experiment.")
    RE(agent.learn(iterations=30))


The final line will run 30 iterations of random search + Bayesian optimization to find the optimal DOFs to satisfy the objective. This will directly control the two EPICS signals and acquire data from the ``outcome1`` signal.
