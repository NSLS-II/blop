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
