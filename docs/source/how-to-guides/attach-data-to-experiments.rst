.. testsetup::
    
    import logging
    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent

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

    dof1 = MovableSignal("dof1")
    dof2 = MovableSignal("dof2")
    dof3 = MovableSignal("dof3")
    readable1 = ReadableSignal("objective1")
    readable2 = ReadableSignal("objective2")

Attach external data to experiments
===================================

In this guide, we will instruct you how to attach external data to an experiment.


Load your data
--------------

We will use fake data for this example. You will be responsible for loading your data from your own source.

.. testcode::

    import pandas as pd

    df = pd.DataFrame({
        "dof1": [1, 2, 3, 4, 5],
        "dof2": [1, 2, 3, 4, 5],
        "dof3": [1, 2, 3, 4, 5],
        "objective1": [1, 2, 3, 4, 5],
        "objective2": [1, 2, 3, 4, 5],
    })

Transform your data to the correct format
-----------------------------------------

.. testcode::

    data = df.to_dict(orient="records")

Configure an agent
------------------

The ``DOF`` and ``Objective`` names must match the keys in the data dictionaries.

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
        Objective(name="objective2", target="min"),
    ]

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        """Replace this with your own evaluation function."""
        outcomes = []
        for suggestion in suggestions:
            outcome = {
                "_id": suggestion["_id"],
                "objective1": 0.1,
                "objective2": 0.2,
            }
            outcomes.append(outcome)
        return outcomes

    agent = Agent(
        readables=[readable1, readable2],
        dofs=dofs,
        objectives=objectives,
        evaluation=evaluation_function,
    )

Ingest your data
----------------

After this, the next time you get a suggestion from the agent it will re-train the model(s) with the new data.

.. code-block:: python

    agent.ingest(data)


(Optional) Configure the generation strategy
--------------------------------------------

If no trials have been run yet, you must configure the generation strategy before summarizing the data.

.. code-block:: python

    agent.ax_client.configure_generation_strategy()

Sanity check the data you attached
----------------------------------

Verify the data you attached is correct.

.. code-block:: python

    agent.ax_client.summarize()
