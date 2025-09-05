from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from blop.integrations.ax.agent import AxAgent
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from blop.bayesian.models import LatentGP
from blop.dofs import DOF
from blop.objectives import Objective
from blop.sim import Beamline


def test_sim_beamline(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(device=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min", transform="log"),
        Objective(name="bl_det_wid_y", target="min", transform="log"),
    ]

    agent = AxAgent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )

    agent.configure_experiment(name="test_ax_agent", description="Test the AxAgent")
    RE(agent.learn(iterations=12, n=1))


def test_plot_objective(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(device=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min", transform="log"),
        Objective(name="bl_det_wid_y", target="min", transform="log"),
    ]

    agent = AxAgent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )
    agent.configure_experiment(name="test_ax_agent", description="Test the AxAgent")
    RE(agent.learn(iterations=12, n=1))

    agent.plot_objective(x_dof_name="bl_kbv_dsv", y_dof_name="bl_kbv_usv", objective_name="bl_det_sum")


def test_generation_strategy_sim_beamline(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(device=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(device=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min", transform="log"),
        Objective(name="bl_det_wid_y", target="min", transform="log"),
    ]

    agent = AxAgent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=db,
    )

    # Create a custom generations strategy that uses the Sobol
    # generator for the first 10 trials, and then uses the LatentGP
    # model for the remaining trials.
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

    agent.configure_experiment(name="test_ax_agent", description="Test the AxAgent")
    agent.set_generation_strategy(generation_strategy)
    RE(agent.learn(iterations=12, n=1))

    df = agent.summarize()
    assert "LatentGP" in df["generation_node"].values
