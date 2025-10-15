import numpy as np
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.registry import Generators
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from blop.ax.agent import Agent
from blop.bayesian.models import LatentGP
from blop.dofs import DOF
from blop.objectives import Objective
from blop.sim.beamline import TiledBeamline


def test_ax_agent(RE, setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min"),
        Objective(name="bl_det_wid_y", target="min"),
    ]

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=setup,
    )

    agent.configure_experiment(name="test_ax_agent", description="Test the Agent")
    RE(agent.learn(iterations=12, n=1))


def test_plot_objective(RE, setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min"),
        Objective(name="bl_det_wid_y", target="min"),
    ]

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=setup,
    )
    agent.configure_experiment(name="test_ax_agent", description="Test the Agent")
    RE(agent.learn(iterations=12, n=1))

    agent.plot_objective(x_dof_name="bl_kbv_dsv", y_dof_name="bl_kbv_usv", objective_name="bl_det_sum")


def test_generation_strategy_sim_beamline(RE, setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_dsh, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbh_ush, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min"),
        Objective(name="bl_det_wid_y", target="min"),
    ]

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=setup,
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

    agent.configure_experiment(name="test_ax_agent", description="Test the Agent")
    agent.set_generation_strategy(generation_strategy)
    RE(agent.learn(iterations=12, n=1))

    df = agent.summarize()
    assert "LatentGP" in df["generation_node"].values


def test_attach_data(setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
    ]

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=setup,
    )
    agent.configure_experiment(name="test_ax_agent", description="Test the Agent")

    data = [
        (
            {
                "bl_kbv_dsv": 0.1,
                "bl_kbv_usv": 0.0,
            },
            {
                "bl_det_sum": 250.0,
            },
        ),
        (
            {
                "bl_kbv_dsv": 1.3,
                "bl_kbv_usv": 1.2,
            },
            {
                "bl_det_sum": 234.0,
            },
        ),
    ]

    agent.attach_data(data)

    agent.configure_generation_strategy()
    df = agent.summarize()
    assert len(df) == 2
    assert np.all(df["bl_kbv_dsv"].values == [0.1, 1.3])
    assert np.all(df["bl_kbv_usv"].values == [0.0, 1.2])
    assert np.all(df["bl_det_sum"].values == [250.0, 234.0])


def test_acquire_baseline(RE, setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(movable=beamline.kbv_dsv, type="continuous", search_domain=(-5.0, 5.0)),
        DOF(movable=beamline.kbv_usv, type="continuous", search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max"),
        Objective(name="bl_det_wid_x", target="min", constraint=(None, "baseline")),
        Objective(name="bl_det_wid_y", target="min", constraint=(None, "baseline")),
    ]

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        db=setup,
    )
    agent.configure_experiment(name="test_ax_agent", description="Test the Agent")
    RE(agent.acquire_baseline())
    agent.configure_generation_strategy()
    df = agent.summarize()
    assert len(df) == 1
    assert df["arm_name"].values[0] == "baseline"
    RE(agent.learn(iterations=6, n=1))
