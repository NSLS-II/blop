.. testsetup::
    
    from typing import Any
    import time
    import logging

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from bluesky.run_engine import RunEngine
    from bluesky.callbacks.tiled_writer import TiledWriter
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

    server = SimpleTiledServer()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db = from_uri(server.uri)
    tiled_writer = TiledWriter(db)
    RE = RunEngine({})
    RE.subscribe(tiled_writer)

    dof1 = MovableSignal("dof1")
    dof2 = MovableSignal("dof2")
    dof3 = MovableSignal("dof3")
    readable1 = ReadableSignal("objective1")
    readable2 = ReadableSignal("objective2")

.. testcleanup::

    # Suppress stdout from server.close() otherwise the doctest will fail
    import os
    import contextlib

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        server.close()

Set outcome constraints relative to a baseline
==============================================

This guide will show you how to acquire a baseline reading for your experiment. This is useful when you are specifying constraints for your objectives and want to compare future outcomes to this baseline.

Configure an agent
------------------

Here we configure an agent with three DOFs and two objectives. The second objective has a constraint that it must be greater than the baseline reading to be considered part of the Pareto frontier.

.. testcode::

    from blop import DOF, Objective
    from blop.ax import Agent

    dofs = [
        DOF(movable=dof1, search_domain=(-5.0, 5.0)),
        DOF(movable=dof2, search_domain=(-5.0, 5.0)),
        DOF(movable=dof3, search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="objective1", target="min"),
        Objective(name="objective2", target="max", constraint=("baseline", None)),
    ]

    agent = Agent(
        readables=[readable1, readable2],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )
    agent.configure_experiment(name="experiment_name", description="experiment_description")

Acquire a baseline reading
--------------------------

To acquire a baseline reading, simply call the ``acquire_baseline`` method. Optionally, you can provide a parameterization which moves the DOFs to specific values prior to acquiring the baseline reading.

.. testcode::

    RE(agent.acquire_baseline())

.. testoutput::
   :hide:

   ...

Verify the baseline reading exists
----------------------------------

.. testcode::

    agent.configure_generation_strategy()
    df = agent.summarize()
    assert len(df) == 1
    assert df["arm_name"].values[0] == "baseline"
