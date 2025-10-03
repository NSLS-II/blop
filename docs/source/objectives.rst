Objectives
++++++++++

Objectives define what you want to optimize in your experiment. They are created using the ``Objective`` class and tell the agent whether to minimize or maximize specific measurements, or enforce constraints on them.

Basic Objective Definition
==========================

An objective requires a ``name`` (which should match an output of your measurement) and either a ``target`` for optimization or a ``constraint`` for feasibility:

.. code-block:: python

    from blop import Objective

    # Maximize beam intensity
    intensity_obj = Objective(name="beam_intensity", target="max")
    
    # Minimize beam width
    width_obj = Objective(name="beam_width", target="min")

The agent will use these objectives to guide the optimization process, suggesting new parameter values that improve the targeted objectives.

Constraints
===========

Constraints define feasible regions for your experiment. Unlike targets (which the agent tries to optimize), constraints define boundaries that must be satisfied for a measurement to be considered valid.

Continuous Constraints
----------------------

For continuous measurements, constraints are defined as tuples of ``(lower_bound, upper_bound)``:

.. code-block:: python

    # Temperature must stay between 20-80 degrees
    temp_constraint = Objective(name="temperature", constraint=(20.0, 80.0))
    
    # Pressure must be at most 100 PSI (no lower bound)
    pressure_constraint = Objective(name="pressure", constraint=(-float('inf'), 100.0))

The agent will only consider parameter combinations that are predicted to satisfy all constraints.

Discrete Constraints
--------------------

For discrete outcomes, constraints are defined as sets of acceptable values:

.. code-block:: python

    # Status must be one of these values
    status_constraint = Objective(name="system_status", constraint={"ok", "ready", "nominal"})

Multiple Objectives
===================

The agent supports multi-objective optimization, allowing you to optimize several competing objectives simultaneously. Simply provide a list of objectives to your agent:

.. code-block:: python

    objectives = [
        Objective(name="beam_intensity", target="max"),     # Want high intensity
        Objective(name="beam_width_x", target="min"),       # Want narrow beam in X
        Objective(name="beam_width_y", target="min"),       # Want narrow beam in Y
        Objective(name="temperature", constraint=(20, 80)), # Temperature constraint
    ]

When multiple optimization targets are present, the agent uses multi-objective optimization strategies to find Pareto-optimal solutions - points where improving one objective would require worsening another.

Usage with Agent
================

Once you've defined your objectives, pass them to the agent along with your degrees of freedom:

.. code-block:: python

    from blop.ax import Agent
    
    agent = Agent(
        readables=[detector1, detector2],
        dofs=[dof1, dof2], 
        objectives=objectives,
        db=databroker_instance,
        digestion=your_digestion_function
    )

The agent automatically converts your blop objectives to the appropriate Ax optimization configuration, handling both single and multi-objective cases transparently.

Best Practices
==============

- **Name Matching**: Ensure objective names exactly match columns produced by your digestion function
- **Constraints vs Targets**: Use constraints for hard boundaries and targets for quantities you want to optimize
- **Balance Objectives**: With multiple targets, consider the trade-offs between competing objectives
- **Start Simple**: Begin with single objectives and add complexity as needed

API Reference
-------------

.. autoclass:: blop.objectives.Objective
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :no-index:
