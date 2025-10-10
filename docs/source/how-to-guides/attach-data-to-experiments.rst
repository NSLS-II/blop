Attach external data to experiments
===================================

In this guide, we will instruct you how to attach external data to an experiment.

Load your data
--------------

We will use fake data for this example. You will be responsible for loading your data from your own source.

.. code-block:: python

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

Configure an agent
------------------

The ``DOF`` and ``Objective`` names must match the keys in the data dictionaries.

.. code-block:: python

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

    agent = Agent(
        readables=[readable1, readable2],
        dofs=dofs,
        objectives=objectives,
        ... # Other arguments
    )
    agent.configure_experiment(name="experiment_name", description="experiment_description")

Attach your data to the experiment
----------------------------------

After this, the next time you get a suggestion from the agent it will re-train the model(s) with the new data.

.. code-block:: python

    agent.attach_data(data)


(Optional) Configure the generation strategy
--------------------------------------------

If no trials have been run yet, you must configure the generation strategy before summarizing the data.

.. code-block:: python

    agent.configure_generation_strategy()

Sanity check the data you attached
----------------------------------

Verify the data you attached is correct.

.. code-block:: python

    agent.summarize()
