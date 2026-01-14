Interactive Optimization
=======================

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

    1. Would you like to go interactively?
       ├─ No: Run optimize_normal (automatic mode)
       │         └─ Completes all iterations automatically
       │
       └─ Yes: Interactive mode
              │
              1. Manually approve suggestions?
                 ├─ No:  Use optimize_step (automated suggestions)
                 │         └─ Go to step 8 (post-iteration options)
                 │
                 └─ Yes:  Manual approval mode
                          │
                          1. How many steps before next approval? (x)
                          │
                          2. For each iteration:
                          |   - Suggest point
                          |   - Ask: "Do you approve this point?"
                          |   - If Yes → evaluate point
                          |   - If No  → abandon point, suggest new one
                          │
                          3. After x iterations complete
                             └─ Go to step 8
    
    2. What would you like to do?
       ├─ c → Continue optimization (no manual suggestions)
       │      └─ Return to step 2
       │
       ├─ s → Suggest points manually
       │      └─ Enter DOF values and objective values
       │         └─ Ingest into model
       │         └─ Return to step 2
       │
       └─ q → Quit optimization

Starting an Interactive Optimization
-------------------------------------

To start an interactive optimization, simply run the ``optimize`` method after defining your agent:

.. code-block:: python

    RE(agent.optimize(iterations=10, n_points=1))

Initial Prompt
~~~~~~~~~~~~~~

When you start the optimization, you'll see:

.. code-block:: text

    +----------------------------------------------------------+
    | Would you like to run the optimization in interactive    |
    | mode?                                                    |
    +----------------------------------------------------------+
      y: Yes
      n: No
    
    Enter choice [y,n]:

- Choose ``y`` for interactive mode with full control
- Choose ``n`` for automatic mode (runs all iterations without prompts)

Automatic Mode (Non-Interactive)
---------------------------------

If you choose ``n`` (No) at the initial prompt, the optimization runs in automatic mode:

- All iterations execute without user intervention
- Points are suggested, evaluated, and ingested automatically

Manual Approval Mode
--------------------

If you choose ``y`` (Yes) for interactive mode, you'll be asked:

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
    > 

Enter how often you would like to be able to manually approve a suggested point. If for a suggested point, manual approval is given, you'll see:

.. code-block:: text

    Do you approve this point {'x1': 2.34, 'x2': -1.56, '_id': 5}? (y/n): 

**Note:** The first 5 points are automatically approved to build an initial model.

- Enter ``y`` to evaluate this point
- Enter ``n`` to abandon this point (it won't be evaluated, and will be marked as ``abandoned``)

Automated Suggestions (No Manual Approval)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you choose ``n`` (No) for manual approval, the optimizer will:

1. Generate suggestions automatically
2. Evaluate all points without asking for approval
3. Proceed to post-iteration options (see below)

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

When you choose this option, you'll be prompted to enter values for each DOF and objective:

.. code-block:: text

    Enter value for x1 (float): 2.5
    Enter value for x2 (float): 1.3
    Enter value for my_objective (float): 42.7
    
    +----------------------------------------------------------+
    | Do you want to suggest another point?                    |
    +----------------------------------------------------------+
      y: Yes
      n: No, finish suggestions
    
    Enter choice [y,n]:

If you enter an invalid number (like "abc"), you'll see:

.. code-block:: text

    Invalid input. Please enter a valid number for x1.

And you'll be asked to try again for that specific parameter.

Option q: Quit Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ends the optimization immediately

See Also
--------

- :doc:`/tutorials/simple-experiment` - Basic optimization tutorial
