Agent
+++++

The Agent is the central component that coordinates optimization, data acquisition, and experiment tracking. It combines your DOFs and objectives with Bluesky's data collection capabilities to automatically explore and optimize your experimental space.

Basic Agent Setup
=================

To create an agent, you need to provide DOFs (parameters to control), objectives (what to optimize), readables (detectors to measure), a databroker instance, and optionally a digestion function:

.. code-block:: python

    from blop.ax import Agent
    from blop import DOF, Objective

    # Define your experimental parameters
    dofs = [
        DOF(name="motor_x", search_domain=(-5.0, 5.0)),
        DOF(device=motor_y, search_domain=(0.0, 10.0)),
    ]

    # Define what to optimize
    objectives = [
        Objective(name="intensity", target="max"),
        Objective(name="width", target="min"),
    ]

    # Create the agent
    agent = Agent(
        readables=[detector1, detector2],
        dofs=dofs,
        objectives=objectives,
        db=databroker_instance,
        digestion=your_digestion_function
    )

The agent uses Ax (Meta's Bayesian optimization platform) under the hood to intelligently suggest parameter combinations and track experiment results. By default, the Bluesky measurement plan will be

.. code-block:: python

    from bluesky.plans import list_scan

    list_scan([detector1, detector2], motor_x, [x1, ...], motor_y, [y1, ...])

Where `x1` and `y1` are the suggested parameter values to try for the `motor_x` and `motor_y` DOFs, respectively. See `list_scan <https://blueskyproject.io/bluesky/main/generated/bluesky.plans.list_scan.html#bluesky-plans-list-scan>`_ for more details.

Experiment Configuration
========================

Before running optimization, configure your experiment with metadata:

.. code-block:: python

    agent.configure_experiment(
        name="beam_optimization",
        description="Optimize beam focus and intensity",
        experiment_type="optimization"
    )

This sets up the underlying Ax experiment with your DOFs as parameters and your objectives as targets or constraints.

Data Processing with Digestion
==============================

The digestion function processes raw data from your detectors into objective values. It receives the trial index, active objectives, and a pandas DataFrame of collected data:

.. code-block:: python

    def my_digestion(trial_index, objectives, df):
        """Process detector data into objective values."""
        # Extract relevant data for this trial
        trial_data = df.iloc[trial_index]
        
        # Compute objective values from raw measurements
        intensity = trial_data['detector1_sum']
        width = trial_data['detector2_width']
        
        # Return objectives as {name: (value, std_error)} pairs
        return {
            'intensity': (intensity, None),  # None means no error estimate
            'width': (width, 0.1)  # Include error estimate if available
        }

If no digestion function is provided, the agent assumes objective names match column names in the data and that the data are error-free.

Running Optimization
====================

The agent provides several ways to run optimization:

Automatic Optimization Loop
---------------------------

The simplest approach runs complete optimization cycles automatically:

.. code-block:: python

    # Run 5 iterations with 4 trials each
    RE(agent.optimize(iterations=5, n=4))

This automatically suggests parameters, runs experiments, and ingests results.

Manual Control
--------------

For more control, you can manually manage the optimization cycle:

.. code-block:: python

    # Get suggested parameter values
    trials = agent.suggest(n=3)
    
    # Run experiments (this returns data for ingestion)
    # You can deploy your own trials in any way you wish here
    data = yield from agent.acquire(trials)
    
    # Feed results back to the optimizer
    agent.ingest(trials, data)

Attaching Existing Data
-----------------------

You can initialize the agent with existing experimental data:

.. code-block:: python

    # Previous experimental results
    existing_data = [
        ({"motor_x": 1.0, "motor_y": 2.0}, {"intensity": (100.0, 5.0)}),
        ({"motor_x": -1.0, "motor_y": 3.0}, {"intensity": (80.0, 4.0)}),
    ]
    
    agent.attach_data(existing_data)

This helps the agent start with prior knowledge instead of random exploration.

Analysis and Visualization
==========================

The agent provides built-in analysis capabilities:

Objective Visualization
-----------------------

Plot how objectives vary across the parameter space:

.. code-block:: python

    # Plot objective as function of two DOFs
    agent.plot_objective(
        x_dof_name="motor_x", 
        y_dof_name="motor_y", 
        objective_name="intensity"
    )

Custom Analysis
---------------

Compute custom analyses using Ax's analysis framework:

.. code-block:: python

    from ax.analysis import ContourPlot
    
    # Compute custom analysis
    analysis_cards = agent.compute_analyses([
        ContourPlot(x_parameter_name="motor_x", 
                   y_parameter_name="motor_y", 
                   metric_name="intensity")
    ])

Experiment Tracking
===================

The agent automatically tracks all trials and results. You can view the current state:

.. code-block:: python

    # Get summary of all trials
    summary_df = agent.summarize()
    print(summary_df)

This shows all parameter combinations tried and their resulting objective values.

Generation Strategy
===================

The agent can be configured with different optimization strategies:

.. code-block:: python

    # Configure the optimization approach
    agent.configure_generation_strategy(
        initialization_budget=10,  # points for initial exploration
        initialize_with_center=True  # start at center of search space
    )

Advanced users can provide custom Ax generation strategies for specialized optimization approaches.

Bluesky Integration
===================

The agent seamlessly integrates with Bluesky:

- **Plans**: Automatically generates ``list_scan`` plans to move DOFs and collect data
- **Metadata**: Tracks trial information in Bluesky run documents  
- **RunEngine**: Works with your existing RE setup and callbacks
- **DataBroker**: Retrieves results automatically after data collection

The agent handles all the details of coordinating between optimization suggestions and experimental execution.

Best Practices
==============

- **Start Simple**: Begin with a few DOFs and one objective, then add complexity
- **Good Digestion**: Ensure your digestion function correctly maps detector data to objectives  
- **Reasonable Bounds**: Set DOF search domains that cover meaningful parameter ranges
- **Monitor Progress**: Use visualization tools to understand optimization behavior
- **Save State**: The agent maintains full experiment history for reproducibility
- **Error Handling**: The agent is designed to be robust to occasional experimental failures
