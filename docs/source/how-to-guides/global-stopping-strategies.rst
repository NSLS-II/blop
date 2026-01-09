.. testsetup::
    
    from typing import Any
    import time
    import logging

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from bluesky.run_engine import RunEngine
    from bluesky.callbacks.tiled_writer import TiledWriter
    from bluesky.callbacks.best_effort import BestEffortCallback

    from tiled.client.container import Container
    from tiled.client import from_uri
    from tiled.server import SimpleTiledServer

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

    # Start a local Tiled server for data storage
    tiled_server = SimpleTiledServer()

    # Set up the Bluesky RunEngine and connect it to Tiled
    RE = RunEngine({})
    tiled_client = from_uri(tiled_server.uri)
    tiled_writer = TiledWriter(tiled_client)
    RE.subscribe(tiled_writer)
    bec = BestEffortCallback()
    bec.disable_plots()
    RE.subscribe(bec)

    x1 = MovableSignal("x1", initial_value=0.1)
    x2 = MovableSignal("x2", initial_value=0.23)

.. testcleanup::

    # Suppress stdout from server.close() otherwise the doctest will fail
    import os
    import contextlib

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        tiled_server.close()

Using a global stopping strategy
==================================
This guide will show you how to use a global stopping strategy. This allows you to stop an optimization early based on certain criteria, such as lack of improvement over a series of trials.

Define the stopping strategy    
----------------------------
You will need to define the following parameters: 
1. The minimum number of trials `min_trials` before checking for improvement
2.  The window size `window_size`, how many of the most recent trials to consider when checking for improvement
3.  The improvement bar `improvement_bar`, the theshold for considering improvement relative to the interquartile range of values seen so far. Must be >= 0

.. testcode::

    from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy

    stopping_strategy = ImprovementGlobalStoppingStrategy(
        min_trials=10,
        window_size=5,
        improvement_bar=0.1,
    )

Configure an agent
------------------

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective

    class Himmelblau2DEvaluation():
        def __init__(self, tiled_client: Container):
            self.tiled_client = tiled_client

        def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
            run = self.tiled_client[uid]
            outcomes = []
            x1_data = run["primary/x1"].read()
            x2_data = run["primary/x2"].read()

            for suggestion in suggestions:
                suggestion_id = suggestion["_id"]
                x1 = x1_data[suggestion_id % len(x1_data)]
                x2 = x2_data[suggestion_id % len(x2_data)]
                # Himmelblau function
                outcomes.append({
                    "himmelblau_2d": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2,
                    "_id": suggestion_id
                })
            
            return outcomes

    dofs = [
        RangeDOF(actuator=x1, bounds=(-5.0, 5.0), parameter_type="float"),
        RangeDOF(actuator=x2, bounds=(-5.0, 5.0), parameter_type="float"),
    ]

    objectives = [
        Objective(name="himmelblau_2d", minimize=False),
    ]

    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        stopping_strategy=stopping_strategy,
        evaluation=Himmelblau2DEvaluation(tiled_client),
    )

Run the experiment with Bluesky
-------------------------------
The experiment will stop early only if the stopping criteria are met. Otherwise, it will continue for the full number of iterations.
.. testcode::

    RE(agent.optimize(iterations=10000, n_points=1))
