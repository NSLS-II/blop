Manual Point Injection
======================

This guide shows how to inject custom parameter combinations based on domain knowledge or external sources, alongside optimizer-driven suggestions.

Basic Usage
-----------

To evaluate manually-specified points, use the ``sample_suggestions`` method with parameter combinations (without ``"_id"`` keys). The optimizer will automatically register these trials and incorporate the results into the Bayesian model.

.. code-block:: python

    from blop.ax import Agent, RangeDOF, Objective
    
    # Configure agent
    agent = Agent(
        sensors=[sensor],
        dofs=[
            RangeDOF(actuator=motor_x, bounds=(-10, 10)),
            RangeDOF(actuator=motor_y, bounds=(-10, 10)),
        ],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=my_evaluation_function,
    )
    
    # Define points of interest
    manual_points = [
        {'motor_x': 0.5, 'motor_y': 1.0},  # Center region
        {'motor_x': 0.0, 'motor_y': 0.0},  # Origin
    ]
    
    # Evaluate them
    RE(agent.sample_suggestions(manual_points))

The manual points will be treated just like optimizer suggestions - they'll be tracked, evaluated, and used to improve the model.

Mixed Workflows
---------------

You can combine optimizer suggestions with manual points throughout your optimization:

.. code-block:: python

    # Run optimizer for initial exploration
    RE(agent.optimize(iterations=10))
    
    # Try a manual point based on domain insight
    manual_point = [{'motor_x': 0.75, 'motor_y': 0.25}]
    RE(agent.sample_suggestions(manual_point))
    
    # Continue optimization
    RE(agent.optimize(iterations=10))

The optimizer will incorporate your manual point into its model and use it to inform future suggestions.

Manual Approval Workflow
-------------------------

You can review optimizer suggestions before running them by using ``suggest()`` to get suggestions without acquiring data:

.. code-block:: python

    # Get suggestions without running
    suggestions = agent.suggest(n=5)
    
    # Review and filter
    print("Reviewing suggestions:")
    for s in suggestions:
        trial_id = s['_id']
        x = s['motor_x']
        y = s['motor_y']
        print(f"  Trial {trial_id}: x={x:.2f}, y={y:.2f}")
    
    # Only run approved suggestions
    approved = [s for s in suggestions if s['motor_x'] > 0.5]
    
    if approved:
        RE(agent.sample_suggestions(approved))
    else:
        print("No suggestions approved")

This workflow allows you to apply safety checks, domain constraints, or other validation before running trials.

Iterative Refinement
--------------------

A common pattern is to alternate between automated optimization and targeted manual exploration:

.. code-block:: python

    for cycle in range(3):
        # Automated exploration
        RE(agent.optimize(iterations=5, n_points=2))
        
        # Review results and manually probe interesting regions
        # (Look at plots, current best, etc.)
        
        # Try edge cases or special points
        if cycle == 1:
            # After first cycle, check boundaries
            boundary_points = [
                {'motor_x': -10, 'motor_y': 0},
                {'motor_x': 10, 'motor_y': 0},
            ]
            RE(agent.sample_suggestions(boundary_points))

See Also
--------

- :meth:`blop.ax.Agent.suggest` - Get optimizer suggestions without running
- :meth:`blop.ax.Agent.sample_suggestions` - Evaluate specific suggestions
- :meth:`blop.ax.Agent.optimize` - Run full optimization loop
- :class:`blop.protocols.CanRegisterSuggestions` - Protocol for manual trial support
