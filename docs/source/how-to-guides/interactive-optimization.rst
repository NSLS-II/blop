Interactive Optimization
========================

This guide explains how to use Blop's interactive optimization mode, which gives you fine-grained control over the optimization process through user prompts and manual interventions.

Overview
--------

Interactive optimization allows you to:

- Manually approve suggested points before they are evaluated
- Suggest custom points with known objective values during optimization
- Decide when to continue or stop the optimization

The Interactive Optimization Flow
----------------------------------

When you run an optimization, the system follows this workflow:

.. code-block:: text
        1. Number of optimization iterations?
        2. Number of points to suggest per iteration?
        3. Manually approve suggestions?
        |   ├─ No:  Use optimize_step (automated suggestions)
        |   │         └─ Go to step 2 (post-iteration options)
        |   │
        |   └─ Yes:  Manual approval mode
        |            │
        |            a. How many steps before next approval? (x)
        |            │
        |            b. For each iteration:
        |            |   - Suggest point
        |            |   - Ask: "Do you approve this point?"
        |            |   - If Yes → evaluate point
        |            |   - If No  → abandon point, suggest new one
        |            │
        |            c. After x iterations complete
        |               └─ Go to step 4
        |
        4. What would you like to do?
          ├─ c: Continue optimization (no manual suggestions)
          │      └─ Return to step 3
          │
          ├─ s: Suggest points manually
          │      └─ Enter DOF values and objective values
          │         └─ Ingest into model
          │         └─ Return to step 3
          │
          └─ q: Quit optimization

Starting an Interactive Optimization
-------------------------------------

To start an interactive optimization, simply run the ``optimize`` method after defining your agent:

.. code-block:: python

    RE(agent.optimize_interactively())

Initial Prompt
~~~~~~~~~~~~~~

When you start the optimization, you'll see the following prompts:

.. code-block:: text

    +----------------------------------------------------------+
    | Number of optimization iteration                         |
    +----------------------------------------------------------+

- Input the number of iteractions you would like

.. code-block:: text

    +----------------------------------------------------------+
    | Number of points to suggest per iteraction               |
    +----------------------------------------------------------+

- Input the number of points per iteration you want

Manual Approval Mode
--------------------
Afterwards, for every loop of optimization you will see the following:

.. code-block:: text

    +----------------------------------------------------------+
    | Would you like to manually approve suggestions?          |
    +----------------------------------------------------------+
      y: Yes
      n: No
    
    Enter choice [y,n]:

Choosing Manual Approval
~~~~~~~~~~~~~~~~~~~~~~~~~

If you choose ``y`` (Yes), you'll then be asked:

.. code-block:: text

    +----------------------------------------------------------+
    | Number of steps before next approval                     |
    +----------------------------------------------------------+

Enter how often you would like to be able to manually approve a suggested point. If for a suggested point, manual approval is given, you'll see:

.. code-block:: text

    Do you approve this point {'x1': 2.34, 'x2': -1.56, '_id': 5}? (y/n): 

**Note:** The first 5 points are automatically approved to build an initial model.

- Enter ``y`` to evaluate this point
- Enter ``n`` to abandon this point (it won't be evaluated, and will be marked as ``abandoned``)

Post-Iteration Options
----------------------

After completing the iteration(s), you'll be prompted:

.. code-block:: text

    +----------------------------------------------------------+
    | What would you like to do?                               |
    +----------------------------------------------------------+
      c: continue optimization without suggestion
      s: suggest points manually
      q: quit optimization
    
    Enter choice [c,s,q]:

Option c: Continue Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continues with another round of optimization iterations. The system will ask again if you want manual approval for the next round.

Use this when:

- You want to run more iterations
- You're satisfied with the current model and want to let it explore more

Option s: Suggest Points Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allows you to manually input points with known objective values. This is useful when:

- You have external data to add to the model
- You want to guide the optimization to specific regions
- You've performed experiments outside of Blop and want to incorporate the results

When you choose this option, you'll be prompted to enter values for each DOF as a list of dictionaries with the keys as the DOFs

Option q: Quit Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ends the optimization immediately

See Also
--------

- :doc:`/tutorials/simple-experiment` - Basic optimization tutorial
