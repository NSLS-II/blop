from ax import (
    ComparisonOp,
    Models,
    Objective,
    OptimizationConfig,
    OutcomeConstraint,
    ParameterType,
    RangeParameter,
    SearchSpace,
)

from blop.integrations.ax import BlopExperiment, DatabrokerMetric
from blop.sim import Beamline
from blop.utils import get_beam_stats


def test_ax_experiment(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    parameters = [
        RangeParameter(name="bl_kbv_dsv", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
        RangeParameter(name="bl_kbv_usv", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
        RangeParameter(name="bl_kbh_dsh", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
        RangeParameter(name="bl_kbh_ush", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
    ]
    search_space = SearchSpace(parameters=parameters)

    image_sum_metric = DatabrokerMetric(
        broker=db,
        name="bl_det_sum",
        param_names=["bl_det_image"],
        compute_fn=lambda df: (df["bl_det_image"].sum().mean(), 0.0),
    )
    optimization_config = OptimizationConfig(
        objective=Objective(metric=image_sum_metric, minimize=False),
        outcome_constraints=[
            OutcomeConstraint(
                metric=DatabrokerMetric(
                    broker=db,
                    param_names=["bl_det_image"],
                    name="bl_det_wid_x",
                    compute_fn=lambda df: (get_beam_stats(df["bl_det_image"].iloc[0], threshold=0.0)["wid_x"], 0.0),
                ),
                op=ComparisonOp.LEQ,
                bound=10,
                relative=False,
            ),
            OutcomeConstraint(
                metric=DatabrokerMetric(
                    broker=db,
                    param_names=["bl_det_image"],
                    name="bl_det_wid_y",
                    compute_fn=lambda df: (get_beam_stats(df["bl_det_image"].iloc[0], threshold=0.0)["wid_y"], 0.0),
                ),
                op=ComparisonOp.LEQ,
                bound=10,
                relative=False,
            ),
        ],
    )

    readables = [beamline.det]
    movables = [beamline.kbv_dsv, beamline.kbv_usv, beamline.kbh_dsh, beamline.kbh_ush]

    experiment = BlopExperiment(
        RE=RE,
        readables=readables,
        movables=movables,
        name="test_ax_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
    )

    sobol = Models.SOBOL(experiment.search_space)
    for _ in range(5):
        trial = experiment.new_trial(generator_run=sobol.gen(1))
        # TODO: Try RE(trial.run())
        trial.run()
        trial.mark_completed()

    best_arm = None
    for _ in range(5):
        gpei = Models.BOTORCH_MODULAR(experiment=experiment, data=experiment.fetch_data())
        generator_run = gpei.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        trial = experiment.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    experiment.fetch_data()
    assert best_arm is not None
