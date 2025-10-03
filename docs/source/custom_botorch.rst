Custom BoTorch Models
=====================

Blop supports custom BoTorch models and acquisition functions through Ax's flexible generation strategy system. This allows you to use specialized models like the built-in ``LatentGP`` or integrate your own custom models for advanced optimization scenarios.

Overview
--------

By default, Blop uses standard Gaussian Process models for Bayesian optimization. However, you can customize the optimization strategy by:

- Using different initialization strategies (Sobol, Latin Hypercube, etc.)
- Switching to specialized models like ``LatentGP`` for high-dimensional problems
- Configuring custom acquisition functions
- Setting custom optimization parameters

The ``LatentGP`` Model
----------------------

The ``LatentGP`` model is a specialized Gaussian Process designed for optimization problems with:

- High-dimensional parameter spaces
- Complex, non-linear relationships between parameters
- Skewed or transformed input dimensions

It uses a custom latent kernel that can better capture complex correlations in the parameter space.

Basic Custom Generation Strategy
---------------------------------

Here's how to create a custom generation strategy that uses Sobol sampling for initialization and then switches to the ``LatentGP`` model:

.. code-block:: python

    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.model_spec import GeneratorSpec
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.modelbridge.registry import Generators
    from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    from blop.bayesian.models import LatentGP
    from blop.ax import Agent

    # Create your standard agent setup (DOFs, objectives, etc.)
    agent = Agent(
        readables=[detector1, detector2],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )

    # Create a custom generation strategy
    generation_strategy = GenerationStrategy(
        name="Custom Generation Strategy",
        nodes=[
            # Phase 1: Use Sobol sampling for initial exploration
            GenerationNode(
                node_name="Sobol",
                model_specs=[
                    GeneratorSpec(model_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
                ],
                transition_criteria=[
                    MinTrials(
                        threshold=10,  # Use Sobol for first 10 trials
                        transition_to="LatentGP",  # Then switch to LatentGP node
                        use_all_trials_in_exp=True,
                    ),
                ],
            ),
            # Phase 2: Use LatentGP for exploitation
            GenerationNode(
                node_name="LatentGP",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "surrogate_spec": SurrogateSpec(
                                model_configs=[
                                    ModelConfig(
                                        botorch_model_class=LatentGP,
                                        input_transform_classes=None,
                                        model_options={},
                                    ),
                                ],
                            ),
                            "botorch_acqf_class": qLogNoisyExpectedImprovement,
                            "acquisition_options": {},
                        },
                        model_gen_kwargs={
                            "optimizer_kwargs": {
                                "num_restarts": 10,  # Number of optimization restarts
                                "sequential": True,   # Sequential optimization
                            },
                        },
                    ),
                ],
            ),
        ],
    )

    # Apply the custom strategy to your agent
    agent.configure_experiment(name="custom_optimization", description="Using LatentGP")
    agent.set_generation_strategy(generation_strategy)

    # Run optimization
    RE(agent.learn(iterations=20, n=1))

Configuration Options
---------------------

Initialization Phase
~~~~~~~~~~~~~~~~~~~~~

You can customize the initial exploration phase:

.. code-block:: python

    # Use Latin Hypercube instead of Sobol
    GeneratorSpec(model_enum=Generators.UNIFORM, model_kwargs={"seed": 42})

    # Adjust the number of initialization trials
    MinTrials(threshold=15, transition_to="LatentGP")

Model Configuration
~~~~~~~~~~~~~~~~~~~

Configure the ``LatentGP`` model with different options:

.. code-block:: python

    ModelConfig(
        botorch_model_class=LatentGP,
        input_transform_classes=None,
        model_options={
            "skew_dims": [(0, 1), (2, 3)],  # Specify skewed dimensions
        },
    )

Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~

Choose different acquisition functions:

.. code-block:: python

    from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement

    # Standard Expected Improvement
    "botorch_acqf_class": qExpectedImprovement,

    # Noisy Expected Improvement (default for LatentGP)
    "botorch_acqf_class": qLogNoisyExpectedImprovement,

Optimization Settings
~~~~~~~~~~~~~~~~~~~~~

Tune the optimization parameters:

.. code-block:: python

    model_gen_kwargs={
        "optimizer_kwargs": {
            "num_restarts": 20,     # More restarts for better optimization
            "sequential": False,    # Parallel optimization
            "raw_samples": 1000,    # Samples for candidate generation
        },
    }

Complete Example
----------------

Here's a complete working example using a simulated beamline:

.. code-block:: python

    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.model_spec import GeneratorSpec
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.modelbridge.registry import Generators
    from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    from blop.bayesian.models import LatentGP
    from blop.dofs import DOF
    from blop.ax import Agent
    from blop.objectives import Objective
    from blop.sim import Beamline

    # Set up a simulated beamline
    beamline = Beamline(name="my_beamline")
    beamline.det.noise.put(False)

    # Define degrees of freedom (parameters to optimize)
    dofs = [
        DOF(device=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    # Define optimization objectives
    objectives = [
        Objective(name="bl_det_sum", target="max"),           # Maximize intensity
        Objective(name="bl_det_wid_x", target="min", transform="log"),  # Minimize X width
        Objective(name="bl_det_wid_y", target="min", transform="log"),  # Minimize Y width
    ]

    # Create agent
    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )

    # Define custom generation strategy
    custom_strategy = GenerationStrategy(
        name="Sobol + LatentGP Strategy",
        nodes=[
            GenerationNode(
                node_name="SobolInitialization",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.SOBOL,
                        model_kwargs={"seed": 0}
                    ),
                ],
                transition_criteria=[
                    MinTrials(threshold=8, transition_to="LatentGPOptimization"),
                ],
            ),
            GenerationNode(
                node_name="LatentGPOptimization",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "surrogate_spec": SurrogateSpec(
                                model_configs=[
                                    ModelConfig(
                                        botorch_model_class=LatentGP,
                                        input_transform_classes=None,
                                        model_options={},
                                    ),
                                ],
                            ),
                            "botorch_acqf_class": qLogNoisyExpectedImprovement,
                        },
                        model_gen_kwargs={
                            "optimizer_kwargs": {
                                "num_restarts": 15,
                                "sequential": True,
                            },
                        },
                    ),
                ],
            ),
        ],
    )

    # Configure and run optimization
    agent.configure_experiment(
        name="custom_latent_gp_optimization",
        description="Multi-objective optimization with LatentGP"
    )
    agent.set_generation_strategy(custom_strategy)

    # Execute optimization
    RE(agent.learn(iterations=25, n=1))

    # Analyze results
    summary_df = agent.summarize()
    print(f"Used models: {summary_df['generation_node'].unique()}")

    # Plot results
    agent.plot_objective(
        x_dof_name="bl_kbv_dsv",
        y_dof_name="bl_kbv_usv",
        objective_name="bl_det_sum"
    )

When to Use Custom Models
-------------------------

Custom models are most beneficial when the default Gaussian Process models in Ax don't capture the underlying physics or structure of your optimization problem. This typically occurs in scenarios where:

**Domain-Specific Knowledge**: You have prior knowledge about the physical relationships between your parameters that can be encoded into a specialized model. For example, if you know certain parameters interact multiplicatively rather than additively, or if there are known symmetries in your system.

**Non-Standard Noise Models**: Your measurements have noise characteristics that deviate from the standard Gaussian assumption, such as heteroscedastic noise that varies across the parameter space, or systematic biases in certain regions.

**Specialized Acquisition Functions**: You need acquisition functions that aren't available in Ax's default set, perhaps ones that incorporate additional constraints, risk measures, or domain-specific utility functions.

**Computational Constraints**: You need models that are specifically optimized for your computational budget, such as sparse GPs for large datasets or models that can leverage GPU acceleration effectively.

**Multi-Fidelity Optimization**: Your experiments involve measurements at different fidelities (e.g., quick simulations vs. expensive real measurements), requiring specialized multi-fidelity models.

Before implementing a custom model, consider whether your needs can be met by:

- Adjusting the default generation strategy parameters
- Using different initialization strategies
- Modifying the acquisition function parameters
- Preprocessing your data or objectives

Custom models require significant expertise in both BoTorch and your specific domain. The investment is typically worthwhile when you have exhausted standard approaches and have clear evidence that domain-specific modeling will provide substantial improvements.

Troubleshooting
---------------

If you encounter issues:

1. **Model fails to train**: Reduce the number of optimization restarts or try a different acquisition function
2. **Poor optimization performance**: Increase the initialization budget or adjust the transition threshold
3. **Memory issues**: Reduce ``raw_samples`` in the optimizer kwargs
4. **Slow optimization**: Set ``sequential=False`` for parallel candidate optimization

For debugging, you can inspect which generation node is being used:

.. code-block:: python

    summary_df = agent.summarize()
    print(summary_df[['generation_node', 'trial_index']])

You can also utilize analyses to help you understand the performance of your custom model. See :doc:`agent` for more details.
