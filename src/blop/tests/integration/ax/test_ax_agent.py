from blop.ax.agent import Agent
from blop.dofs import DOF
from blop.objectives import Objective
from blop.plans import optimize
from blop.sim.beamline import TiledBeamline


def test_ax_agent_sim_beamline(RE, setup):
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

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        run = setup[uid]

        bl_det_sums = run["primary/bl_det_sum"].read()
        bl_det_wid_x = run["primary/bl_det_wid_x"].read()
        bl_det_wid_y = run["primary/bl_det_wid_y"].read()

        trial_ids = run.metadata["start"]["blop_suggestion_ids"]
        outcomes = []
        for suggestion in suggestions:
            idx = trial_ids.index(suggestion["_id"])
            outcome = {
                "_id": suggestion["_id"],
                "bl_det_sum": bl_det_sums[idx],
                "bl_det_wid_x": bl_det_wid_x[idx],
                "bl_det_wid_y": bl_det_wid_y[idx],
            }
            outcomes.append(outcome)

        return outcomes

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        evaluation=evaluation_function,
    )
    RE(optimize(agent.to_optimization_problem(), iterations=12, n_points=1))
