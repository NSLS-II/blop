import datetime
import json  # noqa F401

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import matplotlib as mpl  # noqa F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa F401
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.client import from_uri


RE = RunEngine({})

bec = best_effort.BestEffortCallback()
bec.disable_plots()

RE.subscribe(bec)

#converted from Broker
SERVER_HOST_LOCATION = "http://localhost:8000"
tiled_client = from_uri(SERVER_HOST_LOCATION, api_key = "secret")
tiled_writer = TiledWriter(tiled_client)

RE.subscribe(tiled_writer)

print(tiled_writer)