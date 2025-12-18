from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.transforms.outcome import Log

from blop.bayesian.models import LatentGP

default_generation_strategy = GenerationStrategy(
    name="Custom Generation Strategy",
    nodes=[
        GenerationNode(
            node_name="Sobol",
            model_specs=[
                GeneratorSpec(model_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
            ],
            transition_criteria=[
                MinTrials(
                    threshold=16,
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
                                    outcome_transform_classes=[Log],
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
