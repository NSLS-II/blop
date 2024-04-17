Agent
+++++

The blop ``Agent`` takes care of the entire optimization loop, from data acquisition to model fitting.

.. code-block:: python

    from blop import DOF, Objective, Agent

    dofs = [
        DOF(name="x1", description="the first DOF", search_domain=(-10, 10))
        DOF(name="x2", description="another DOF", search_domain=(-5, 5))
        DOF(name="x3", description="yet another DOF", search_domain=(0, 1))
    ]

    objective = [
        Objective(name="y1", description="something to minimize", target="min")
        Objective(name="y2", description="something to maximize", target="max")
    ]

    dets = [
        my_detector, # an ophyd device with a .trigger() method that determines "y1"
        my_other_detector # a detector that measures "y2"
    ]

    agent = Agent(dofs=dofs, objectives=objectives, dets=dets)


This creates an ``Agent`` with no data about the world, and thus no way to model it.
We have to start with asking the ``Agent`` to learn by randomly sampling the parameter space.
The ``Agent`` learns with Bluesky plans emitted by the ``agent.learn()`` method, which can be passed to a ``RunEngine``:

.. code-block:: python

    RE(agent.learn("qr", n=16)) # the agent chooses 16 quasi-random points, samples them, and fits models to them
