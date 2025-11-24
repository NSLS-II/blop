from blop.ax.agent import Agent
from blop.dofs import DOF
from blop.evaluation import TiledEvaluationFunction
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

    agent = Agent(
        readables=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        evaluation=TiledEvaluationFunction(
            tiled_client=setup,
            objectives=objectives,
        ),
    )
    RE(optimize(agent.to_optimization_problem(), iterations=12, n_points=1))
