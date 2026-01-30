import logging
import time
from typing import Any

from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks import LivePlot
from bluesky.callbacks.core import CallbackBase
from bluesky.callbacks.mpl_plotting import initialize_qt_teleporter
from event_model import RunRouter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from blop import Agent, RangeDOF, Objective

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)


class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)
    def exception(self, timeout = 0.0):
        return None
    @property
    def done(self) -> bool:
        return True
    @property
    def success(self) -> bool:
        return True
   
class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0
    @property
    def name(self) -> str:
        return self._name
    @property
    def hints(self) -> Hints:
        return {"fields": [self._name], "dimensions": [], "gridding": "rectilinear"}
    @property
    def parent(self) -> Any | None:
        return None
    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}
    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}

class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value
    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


server = SimpleTiledServer()
tiled_client = from_uri(server.uri)

RE = RunEngine({})

def callback_factory(name, doc):
    if name == "start" and doc.get("run_key") == "optimize":
        bec = BestEffortCallback()
        tw = TiledWriter(tiled_client)
        return [bec, tw], []
    elif name == "start" and doc.get("run_key") == "default_acquire":
        tw = TiledWriter(tiled_client)
        return [tw], []
    else:
        return [], []

rr = RunRouter([callback_factory])
RE.subscribe(rr)

x1 = MovableSignal("x1")
x2 = MovableSignal("x2")

dofs = [
    RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
    RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
]
objectives = [
    Objective(name="objective1", minimize=True),
    Objective(name="objective2", minimize=True),
]
sensors = []

def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
    return [{"_id": suggestion["_id"], "objective1": 0.1 * (i + 1), "objective2": 0.2 * (i + 1)} for i, suggestion in enumerate(suggestions)]

agent = Agent(sensors=sensors, dofs=dofs, objectives=objectives, evaluation_function=evaluation_function)
