=====
Usage
=====

Working in the Bluesky environment, we need to pass four ingredients to the Bayesian agent:

* ``dofs``: A list of degrees of freedom for the agent.
* ``dets`` (Optional): A list of detectors to be triggered during acquisition.
* ``tasks``: A list of tasks for the agent to maximize.
* ``digestion``: A function that processes the output of the acquisition into the task values.

.. code-block:: python

    import bloptools

    dofs = [
        {"device": some_motor, "limits": (-0.5, 0.5), "kind": "active"},
        {"device": another_motor, "limits": (-0.5, 0.5), "kind": "active"},
    ]

    tasks = [
        {"key": "flux", "kind": "maximize", "transform": "log"}
        ]

    agent = bloptools.bayesian.Agent(
        dofs=dofs,
        tasks=tasks,
        dets=[some_detector, another_detector],
        digestion=your_digestion_function,
        db=db,
    )

    RE(agent.initialize("qr", n_init=24))
