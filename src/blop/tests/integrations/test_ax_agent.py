from blop.dofs import DOF
from blop.integrations.ax.agent import AxAgent
from blop.objectives import Objective
from blop.sim import Beamline


def test_ax_agent(RE, db):
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
    RE(agent.learn(iterations=25, n=1))
