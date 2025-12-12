How to optimize with ophyd and ophyd-async devices
==================================================

This guide will walk you through the process of setting up Blop to optimize with `ophyd <https://blueskyproject.io/ophyd/>`_ and `ophyd-async <https://blueskyproject.io/ophyd-async/>`_ devices.

ophyd devices
-------------

Ophyd's ``Signal`` class implements both the ``Readable`` and ``NamedMovable`` protocols, so they can be used directly with Blop. You can also use the ``SignalRO`` class which only implements the ``Readable`` protocol if you
want to capture this data at each step of the experiment.

The ``name`` attribute of the signal will be used as the name of the :class:`blop.ax.DOF` on the backend.

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    from ophyd import Signal, SignalRO

    some_control_signal = Signal(name="some_control_signal")
    some_readable_signal = SignalRO(name="some_readable_signal")
    dof = RangeDOF(movable=some_control_signal, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        readables=[some_readable_signal],
        dofs=[dof],
        objectives=[Objective(name="result", minimize=False)],
        evaluation=lambda uid, suggestions: [{"result": 0.1}],
    )

ophyd-async devices
-------------------

Ophyd-async's ``SignalW`` class implements the ``NamedMovable`` protocol, so they can also be used directly with Blop. The ``SignalR`` class implements the ``Readable`` protocol. And the ``SignalRW`` implements both.

Below we create *soft* signals that return instances of the above classes.

Once again, the ``name`` attribute of the signal will be used as the name of the :class:`blop.ax.DOF` on the backend.

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    from ophyd_async.core import soft_signal_rw, soft_signal_r_and_setter

    some_control_signal = soft_signal_rw(float, name="some_control_signal")
    some_readable_signal = soft_signal_r_and_setter(float, name="some_readable_signal")
    dof = RangeDOF(movable=some_control_signal, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        readables=[some_readable_signal],
        dofs=[dof],
        objectives=[Objective(name="result", minimize=False)],
        evaluation=lambda uid, suggestions: [{"result": 0.1}],
    )

Using your devices in custom acquisition plans
----------------------------------------------

If you use a custom acquisition plan by implementing the :class:`blop.protocols.AcquisitionPlan` protocol, you can use the ``movables`` and/or ``readables`` arguments to access the ophyd or ophyd-async devices you configured as DOFs.

.. testcode::

    import bluesky.plan_stubs as bps
    from bluesky.protocols import NamedMovable, Readable
    from bluesky.utils import MsgGenerator
    from bluesky.run_engine import RunEngine
    from ophyd_async.core import soft_signal_rw

    from blop.ax import Agent, RangeDOF, Objective
    from blop.protocols import AcquisitionPlan

    def custom_acquire(suggestions: list[dict], movables: list[NamedMovable], readables: list[Readable]) -> MsgGenerator[str]:
        assert movables[0].name == "signal1"
        assert readables[0].name == "signal2"
        yield from bps.null()

    RE = RunEngine({})

    signal1 = soft_signal_rw(float, name="signal1")
    signal2 = soft_signal_rw(float, name="signal2")

    dof = RangeDOF(movable=signal1, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        readables=[signal2],
        dofs=[dof],
        acquisition_plan=custom_acquire,
        objectives=[Objective(name="result", minimize=False)],
        evaluation=lambda uid, suggestions: [{"result": 0.1, "_id": 0}],
    )

    RE(agent.optimize())
