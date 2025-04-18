import pandas as pd
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from blop.integrations.ax import create_blop_experiment, create_bluesky_evaluator
from blop.sim import Beamline
from blop.utils import get_beam_stats


def test_ax_client_experiment(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    ax_client = AxClient()
    create_blop_experiment(
        ax_client,
        parameters=[
            {
                "movable": beamline.kbv_dsv,
                "type": "range",
                "bounds": [-5.0, 5.0],
            },
            {
                "movable": beamline.kbv_usv,
                "type": "range",
                "bounds": [-5.0, 5.0],
            },
            {
                "movable": beamline.kbh_dsh,
                "type": "range",
                "bounds": [-5.0, 5.0],
            },
            {
                "movable": beamline.kbh_ush,
                "type": "range",
                "bounds": [-5.0, 5.0],
            },
        ],
        objectives={"beam_intensity": ObjectiveProperties(minimize=False), "beam_area": ObjectiveProperties(minimize=True)},
    )

    def evaluate(results_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
        stats = get_beam_stats(results_df["bl_det_image"].iloc[0])
        return {"beam_intensity": (stats["sum"], None), "beam_area": (stats["area"], None)}

    evaluator = create_bluesky_evaluator(
        RE, db, [beamline.det], [beamline.kbv_dsv, beamline.kbv_usv, beamline.kbh_dsh, beamline.kbh_ush], evaluate
    )
    for _ in range(25):
        parameterization, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator(parameterization))

    print(ax_client.generation_strategy.trials_as_df)
