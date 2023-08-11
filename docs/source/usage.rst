=====
Usage
=====

Working in the Bluesky environment, we need to pass four ingredients to the Bayesian agent:

* ``dofs``: A list of degrees of freedom for the agent to optimize over.
* ``tasks``: A list of tasks for the agent to optimize.
* ``digestion``: A function that processes the output of the acquisition into the task values.
* ``dets``: (Optional) A list of detectors to be triggered during acquisition.
* ``acquisition``: (Optional) A Bluesky plan to run for each set of inputs.


Degrees of freedom
++++++++++++++++++

Degrees of freedom (DOFs) are passed as an iterable of dicts, each containing at least the device and set of limits.

.. code-block:: python

    my_dofs = [
        {"device": some_motor, "limits": (lower_limit, upper_limit)},
        {"device": another_motor, "limits": (lower_limit, upper_limit)},
    ]

Here ``some_motor`` and ``another_motor`` are ``ophyd`` objects.



Tasks
+++++

Tasks are what we want our agent to try to optimize (either maximize or minimize). We can pass as many as we'd like:

.. code-block:: python

    my_tasks = [
        {"key": "something_to_maximize", "kind": "maximize"}
        {"key": "something_to_minimize", "kind": "minimize"}
        ]



Digestion
+++++++++

The digestion function is how we go from what is spit out by the acquisition to the actual values of the tasks.

.. code-block:: python

    def my_digestion_function(db, uid):

        products = db[uid].table(fill=True) # a pandas DataFrame

        # for each entry, do some
        for index, entry in products.iterrows():

            raw_output_1 = entry.raw_output_1
            raw_output_2 = entry.raw_output_2

            entry.loc[index, "thing_to_maximize"] = some_fitness_function(raw_output_1, raw_output_2)

        return products


Detectors
+++++++++

Detectors are triggered for each input.

.. code-block:: python

    my_dets = [some_detector, some_other_detector]



Acquisition
+++++++++++

We run this plan for each set of DOF inputs. By default, this just moves the active DOFs to the desired points and triggers the supplied detectors.




Building the agent
++++++++++++++++++

Combining these with a databroker instance will construct an agent.

.. code-block:: python

    import bloptools

    my_agent = bloptools.bayesian.Agent(
        dofs=my_dofs,
        dets=my_dets,
        tasks=my_tasks,
        digestion=my_digestion_function,
        db=db, # a databroker instance
    )

    RE(agent.initialize("qr", n_init=24))


In the example below, the agent will loop over the following steps in each iteration of learning.

#. Find the most interesting point (or points) to sample, and move the degrees of freedom there.
#. For each point, run an acquisition plan (e.g., trigger and read the detectors).
#. Digest the results of the acquisition to find the value of the task.
