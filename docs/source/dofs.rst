Degrees of freedom (DOFs)
+++++++++++++++++++++++++

Degrees of freedom (DOFs) define the parameters that your optimization agent can control or monitor during experiments. They represent the input variables that influence your objectives and determine the search space for optimization.

Basic DOF Definition
====================

A DOF requires either a ``name`` (for virtual parameters) or a ``device`` (for real hardware), plus a ``search_domain`` that defines the optimization bounds:

.. code-block:: python

    from blop import DOF

    # Simple continuous DOF with name
    x_dof = DOF(name="x1", search_domain=(-5.0, 5.0))
    
    # DOF connected to real hardware
    motor_dof = DOF(device=my_motor, search_domain=(10.0, 50.0))

The agent will vary these parameters within their search domains to optimize your objectives.

Continuous DOFs
===============

Continuous DOFs represent parameters that can take any real value within a specified range. This is the most common type for physical devices like motors, voltages, or temperatures:

.. code-block:: python

    # Temperature control
    temp_dof = DOF(name="temperature", search_domain=(20.0, 80.0))
    
    # Motor position
    motor_dof = DOF(device=motor_device, search_domain=(-10.0, 10.0))

The agent will intelligently sample points within these bounds and can interpolate between them to find optimal values.

Discrete DOFs
=============

For parameters that can only take specific discrete values, you can define discrete DOFs using sets:

Binary DOFs
-----------

Binary DOFs have exactly two possible values:

.. code-block:: python

    # Shutter open/closed
    shutter_dof = DOF(name="shutter", search_domain={"open", "closed"})

Ordinal DOFs  
------------

Ordinal DOFs have multiple discrete values with a meaningful order:

.. code-block:: python

    # Gain settings with order
    gain_dof = DOF(name="gain", type="ordinal", search_domain={"low", "medium", "high"})

Categorical DOFs
----------------

Categorical DOFs have multiple discrete values without inherent order:

.. code-block:: python

    # Filter selection
    filter_dof = DOF(name="filter", type="categorical", 
                     search_domain={"red", "green", "blue", "clear"})

The agent will explore all possible discrete values but understands the relationships (or lack thereof) between them.

Read-Only DOFs
==============

Sometimes you want the agent to be aware of parameters it cannot control, such as environmental conditions or diagnostic readings:

.. code-block:: python

    # Monitor beam current (can't control it)
    current_dof = DOF(device=beam_current_monitor, read_only=True)
    
    # Monitor temperature (for modeling purposes)
    temp_monitor = DOF(name="ambient_temp", read_only=True)

Read-only DOFs are included in the agent's models as fixed parameters but are never moved during optimization.

Transforms
==========

For parameters that vary over many orders of magnitude, logarithmic transforms can improve optimization:

.. code-block:: python

    # Intensity varies from 1e-6 to 1e6
    intensity_dof = DOF(name="laser_power", 
                       search_domain=(1e-6, 1e6), 
                       transform="log")

This helps the agent sample more effectively across the full range of values.

Bluesky Integration
===================

DOFs are designed to work seamlessly with the Bluesky ecosystem for experiment control. When you connect a DOF to a hardware device, the agent uses Bluesky protocols to move and read the device:

.. code-block:: python

    # Connect DOF to a Bluesky motor
    from ophyd import EpicsMotor
    
    motor = EpicsMotor("XF:28IDC-OP:1{Slt:MB-Ax:X}Mtr", name="slit_motor")
    slit_dof = DOF(device=motor, search_domain=(-5.0, 5.0))

When the agent optimizes, it automatically:

1. **Generates Bluesky plans** - Uses ``list_scan`` and other Bluesky plans to coordinate device movements
2. **Moves devices safely** - Respects device limits and follows proper motion protocols  
3. **Coordinates with RunEngine** - Integrates with your existing Bluesky setup and metadata collection
4. **Handles readbacks** - Automatically reads device positions and includes them in your data

The DOF device parameter accepts any object that implements Bluesky's ``NamedMovable`` protocol, making it compatible with the full range of Bluesky devices including motors, temperature controllers, voltage sources, and custom devices.

For read-only DOFs, the agent will read device values during data collection but never attempt to move them, making it safe to include diagnostic devices, environmental monitors, or other read-only hardware.

Usage with Agent
================

Once you've defined your DOFs, pass them to the agent along with your objectives:

.. code-block:: python

    from blop.ax import Agent

    dofs = [
        DOF(name="x_position", search_domain=(-5.0, 5.0)),
        DOF(device=motor_y, search_domain=(0.0, 10.0)),
        DOF(name="gain", type="ordinal", search_domain={"low", "medium", "high"}),
        DOF(device=temperature_monitor, read_only=True)
    ]

    agent = Agent(
        readables=[detector1, detector2],
        dofs=dofs,
        objectives=objectives,
        db=databroker_instance,
        digestion=your_digestion_function
    )

The agent automatically converts your blop DOFs to the appropriate Ax parameter configuration, handling continuous ranges, discrete choices, and read-only parameters transparently.

Best Practices
==============

- **Reasonable Bounds**: Set search domains that cover the physically meaningful range without being excessive
- **Transform When Needed**: Use logarithmic transforms for parameters spanning multiple orders of magnitude  
- **Include Context**: Use read-only DOFs for environmental factors that affect your experiment
- **Start Simple**: Begin with a few continuous DOFs and add complexity as needed
- **Physical Limits**: Ensure search domains respect the physical limits of your hardware
