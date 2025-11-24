.. testsetup::
    
    from unittest.mock import MagicMock
    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from tiled.client.container import Container

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

    db = MagicMock(spec=Container)

Set DOF constraints
===================

This guide will show you how to set DOF constraints to refine the search space of your optimization.

These constraints are evaluated by the Ax backend. See the `Ax API documentation <https://ax.readthedocs.io/en/stable/api.html#ax.api.client.Client.configure_experiment>`_ for more information.


Create DOFs and an objective
----------------------------

.. testcode::

    from blop import DOF, Objective

    motor_x = MovableSignal(name="motor_x")
    motor_y = MovableSignal(name="motor_y")
    motor_z = MovableSignal(name="motor_z")

    dof1 = DOF(movable=motor_x, search_domain=(0, 1000))
    dof2 = DOF(movable=motor_y, search_domain=(0, 1000))
    dof3 = DOF(movable=motor_z, search_domain=(0, 1000))

    objective = Objective(name="objective1", target="max")


Set a linear constraint
-----------------------

Constraints are specified as strings that are templated and evaluated for you.

.. testcode::

    from blop import DOFConstraint

    constraint = DOFConstraint(constraint="5 * x1 + 2 * x2 <= 4 * x3", x1=motor_x, x2=motor_y, x3=motor_z)

Configure an agent with DOF constraints
---------------------------------------

.. testcode::

    from blop.ax import Agent
    from blop.evaluation import TiledEvaluationFunction

    agent = Agent(
        readables=[],
        dofs=[dof1, dof2, dof3],
        objectives=[objective],
        evaluation=TiledEvaluationFunction(
            tiled_client=db,
            objectives=[objective],
        ),
        dof_constraints=[constraint],
    )

    optimization_problem = agent.to_optimization_problem()
