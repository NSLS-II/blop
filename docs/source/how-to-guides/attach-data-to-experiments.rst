Attach existing data to experiments
===================================

In this guide, we will instruct you how to attach existing data to an experiment.

Load your data
--------------

We will use fake data for this example.

.. code-block:: python

    import pandas as pd

    df = pd.DataFrame({
        "dof1": [1, 2, 3, 4, 5],
        "dof2": [1, 2, 3, 4, 5],
        "dof3": [1, 2, 3, 4, 5],
        "objective1": [1, 2, 3, 4, 5],
        "objective2": [1, 2, 3, 4, 5],
    })

Create an agent
---------------

.. code-block:: python

    from blop import DOF, Objective
    from blop.ax import Agent
    from ophyd import Signal

    dof1 = Signal(name="dof1", value=0.0)
    dof2 = Signal(name="dof2", value=0.0)
    dof3 = Signal(name="dof3", value=0.0)

    dofs = [
        DOF(movable=dof1, search_domain=(-5.0, 5.0)),
        DOF(movable=dof2, search_domain=(-5.0, 5.0)),
        DOF(movable=dof3, search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="objective1", target="min"),
        Objective(name="objective2", target="min"),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        ... # Other arguments
    )

Transform your data to the correct format
-----------------------------------------

.. code-block:: python

    data = []
    for row in df.iterrows():
        data.append(({
                "dof1": row["dof1"],
                "dof2": row["dof2"],
                "dof3": row["dof3"],
            },
            {
                "objective1": row["objective1"],
                "objective2": row["objective2"],
            }
        ))


Attach your data to the experiment
----------------------------------

.. code-block:: python

    agent.attach_data(data)


The next time you get a suggestion from the agent, it will re-train the model(s) with the new data.


Sanity check the data you attached
----------------------------------

.. code-block:: python

    agent.summarize()

This should show you the data you attached.
