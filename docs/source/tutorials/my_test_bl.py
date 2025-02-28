#!/usr/bin/env python3
"""
Created on Fri Sep 27 10:32:06 2024

@author: rchernikov
"""

import os

os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
import time

from matplotlib import pyplot as plt

from blop import DOF, Agent, Objective
from blop.digestion import beam_stats_digestion
from blop.utils import prepare_re_env as pre
from xrt_beamline import Beamline

plt.ion()

kwargs_re = {"db_type": "temp", "root_dir": pre.DEFAULT_ROOT_DIR}
ret = pre.re_env(**kwargs_re)

RE = ret["RE"]
db = ret["db"]
bec = ret["bec"]

h_opt = 0
dh = 5

R1, dR1 = 40000, 10000
R2, dR2 = 20000, 10000

bec.disable_plots()

beamline = Beamline(name="bl")
time.sleep(1)

dofs = [
    DOF(description="KBV R", device=beamline.kbv_dsv, search_domain=(R1 - dR1, R1 + dR1)),
    DOF(description="KBH R", device=beamline.kbh_dsh, search_domain=(R2 - dR2, R2 + dR2)),
]

objectives = [
    Objective(name="bl_det_sum", target="max", transform="log", trust_domain=(20, 1e12)),
    Objective(
        name="bl_det_wid_x",
        target="min",
        transform="log",
        latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
    Objective(
        name="bl_det_wid_y",
        target="min",
        transform="log",
        latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")],
    ),
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

RE(agent.learn("qr", n=32))
RE(agent.learn("qei", n=16, iterations=4))

RE(agent.go_to_best())

agent.plot_objectives(axes=(0, 1))
