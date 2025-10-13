Using custom generation strategies
==================================

This guide will show you how to use custom generation strategies with GPyTorch, BoTorch, Blop, and Ax.

Configure an agent
------------------

.. code-block:: python

    from blop import DOF, Objective
    from blop.ax import Agent

    dofs = [
        DOF(movable=dof1, search_domain=(-5.0, 5.0)),
        DOF(movable=dof2, search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="objective1", target="max"),
    ]

    agent = Agent(
        readables=[readable1, readable2],
        dofs=dofs,
        objectives=objectives,
        ... # Other arguments
    )

Configure a generation strategy
-------------------------------

The following example shows a generation strategy that uses the Sobol generator for the first 10 trials, and then uses the ``LatentGP`` model for the remaining trials.

For more information on generation strategies, see the `Ax documentation <https://ax.dev/docs/generation_strategy>`_. This is essentially a thin wrapper around the Ax generation strategy API. There are many different options to configure which we will not cover in depth here.

.. note::
    
    The is not part of Ax's backward compatibile API. The ``GenerationStrategy`` may be subject to breaking changes in future versions of Ax.

.. code-block:: python

    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.model_spec import GeneratorSpec
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.modelbridge.registry import Generators
    from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    from blop.bayesian.models import LatentGP


    generation_strategy = GenerationStrategy(
        name="Custom Generation Strategy",
        nodes=[
            GenerationNode(
                node_name="Sobol",
                model_specs=[
                    GeneratorSpec(model_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
                ],
                transition_criteria=[
                    MinTrials(
                        threshold=10,
                        transition_to="LatentGP",
                        use_all_trials_in_exp=True,
                    ),
                ],
            ),
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
                                "num_restarts": 10,
                                "sequential": True,
                            },
                        },
                    ),
                ],
            ),
        ],
    )

Configure the experiment and set the generation strategy
--------------------------------------------------------

.. code-block:: python

    agent.configure_experiment(name="latentgp-generation-strategy", description="LatentGP generation strategy")
    agent.set_generation_strategy(generation_strategy)

Run the experiment with Bluesky
-------------------------------

.. code-block:: python

    RE(agent.learn(iterations=12, n=1))


Verify the generation strategy was used
---------------------------------------

.. code-block:: python

    df = agent.summarize()
    assert "LatentGP" in df["generation_node"].values
