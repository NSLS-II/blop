import numpy as np

from blop import DOF, Agent, Objective
from blop.digestion import beam_stats_digestion
from blop.sim import Beamline


def test_kb_simulation(RE, db):
    beamline = Beamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        DOF(description="KBV downstream", device=beamline.kbv_dsv, search_domain=(-5.0, 5.0)),
        DOF(description="KBV upstream", device=beamline.kbv_usv, search_domain=(-5.0, 5.0)),
        DOF(description="KBH downstream", device=beamline.kbh_dsh, search_domain=(-5.0, 5.0)),
        DOF(description="KBH upstream", device=beamline.kbh_ush, search_domain=(-5.0, 5.0)),
    ]

    objectives = [
        Objective(name="bl_det_sum", target="max", transform="log", trust_domain=(200, np.inf)),
        Objective(name="bl_det_wid_x", target="min", transform="log", latent_groups=[("bl_kbh_dsh", "bl_kbh_ush")]),
        Objective(name="bl_det_wid_y", target="min", transform="log", latent_groups=[("bl_kbv_dsv", "bl_kbv_usv")]),
    ]

    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        detectors=[beamline.det],
        digestion=beam_stats_digestion,
        digestion_kwargs={"image_key": "bl_det_image"},
        verbose=True,
        db=db,
        tolerate_acquisition_errors=False,
        enforce_all_objectives_valid=True,
        train_every=3,
    )

    RE(agent.learn("qr", n=16))
    RE(agent.learn("qei", n=4, iterations=4))
