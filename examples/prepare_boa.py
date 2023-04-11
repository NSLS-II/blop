import datetime
import json  # noqa F401

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker
import matplotlib as mpl  # noqa F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa F401
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree
from sirepo_bluesky.shadow_handler import ShadowFileHandler
from sirepo_bluesky.sirepo_bluesky import SirepoBluesky
from sirepo_bluesky.sirepo_ophyd import create_classes
from sirepo_bluesky.srw_handler import SRWFileHandler

import bloptools  # noqa F401
from bloptools.bo import BayesianOptimizationAgent  # noqa F401

RE = RunEngine({})

bec = best_effort.BestEffortCallback()
bec.disable_plots()

RE.subscribe(bec)

# MongoDB backend:
db = Broker.named("local")  # mongodb backend
try:
    databroker.assets.utils.install_sentinels(db.reg.config, version=1)
except Exception:
    pass

RE.subscribe(db.insert)
db.reg.register_handler("srw", SRWFileHandler, overwrite=True)
db.reg.register_handler("shadow", ShadowFileHandler, overwrite=True)
db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)

plt.ion()

root_dir = "/tmp/sirepo-bluesky-data"
_ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

connection = SirepoBluesky("http://localhost:8001")

data, schema = connection.auth("shadow", "00000002")
classes, objects = create_classes(connection.data, connection=connection)
globals().update(**objects)

data["models"]["simulation"]["npoint"] = 100000
data["models"]["watchpointReport12"]["histogramBins"] = 32
# w9.duration.kind = "hinted"

bec.disable_baseline()
bec.disable_heading()
bec.disable_table()

# This should be done by installing the package with `pip install -e .` or something similar.
# import sys
# sys.path.insert(0, "../")

mi = np.array([0, 1])

dofs = [[kbv.x_rot, kbv.offz, kbh.x_rot, kbh.offz][i] for i in mi]  # noqa F821

hard_bounds = np.array([[-0.20, +0.20], [-1.00, +1.00], [-0.20, +0.20], [-1.00, +1.00]])[mi] * 5e-1

# for dof in dofs:
#    dof.kind = "hinted"
