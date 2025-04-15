from typing import Sequence

from ax import Runner
from ax.core.metric import Metric
from ax.core.trial import Trial
from bluesky import RunEngine
from bluesky.plans import scan
from bluesky.protocols import Readable


class BlopRunner(Runner):
    def __init__(self, RE: RunEngine, detectors: Sequence[Readable], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._RE = RE

    def run(self, trial: Trial, **kwargs):
        # TODO: Match `Moveable`s to their values from the arm
        self._RE(scan(self._detectors, *trial.arm.parameters.values()))


class BlopMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch_trial_data(self, trial: Trial, **kwargs):
        # TODO: Call databroker/tiled to get the data back
        ...
