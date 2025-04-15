from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from ax import Arm, Data, Experiment, Metric, Runner, Trial
from bluesky import RunEngine
from bluesky.plans import list_scan
from bluesky.protocols import HasName, Movable, NamedMovable, Readable
from databroker import Broker
from tiled.client.container import Container


class BlopExperiment(Experiment):
    def __init__(self, RE: RunEngine, readables: Sequence[Readable], movables: Sequence[NamedMovable], *args, **kwargs):
        super().__init__(*args, runner=BlopRunner(RE, readables, movables), **kwargs)
        self._validate_search_space(movables)
        self._validate_optimization_config(readables, movables)

    def _validate_search_space(self, movables: Sequence[NamedMovable]):
        """Validates that the parameters are compatible with the `Movable`s."""
        parameter_names = set(self.search_space.parameters.keys())
        for m, p in zip(movables, self.search_space.parameters.values(), strict=False):
            if m.name != p.name:
                if m.name not in parameter_names:
                    raise ValueError(f"The movable name {m.name} is not a parameter in the search space.")
                raise ValueError(
                    f"The moveable name {m.name} is in the search space, but the order is not correct. "
                    "The order of movables must match the order of the parameters in the search space "
                    "so we can unpack the arm correctly."
                )

    def _validate_optimization_config(self, readables: Sequence[Readable], movables: Sequence[NamedMovable]):
        """Validates that the objectives are compatible with the `Readable`s."""
        # Check that each metric is a BlopMetric
        metrics = self.optimization_config.objective.metrics
        if any(not isinstance(m, BlopMetric) for m in metrics):
            non_blop_metrics = "\n".join([f"{m.name}: {type(m)}" for m in metrics if not isinstance(m, BlopMetric)])
            raise ValueError(f"All objectives must inherit from `BlopMetric`, but found:\n{non_blop_metrics}")

        # Check that each metric's parameters reference a `Readable` or `Movable`
        metric_param_names = {p for m in cast(Sequence[BlopMetric], metrics) for p in m.param_names}
        unmatched_parameters = {
            p
            for p in metric_param_names
            if not any(r.name in p for r in readables) and not any(m.name in p for m in movables)
        }
        if unmatched_parameters:
            raise ValueError(
                f"The following parameters are not referenced in any `Readable` or `Movable`: {unmatched_parameters}"
            )


class BlopRunner(Runner):
    def __init__(self, RE: RunEngine, readables: Sequence[Readable], movables: Sequence[Movable], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._RE = RE
        self._readables = readables
        self._movables = movables

    def _unpack_arm(self, arm: Arm) -> list[Movable | list[Any]]:
        """Unpacks the arm's parameters into the format of the `list_scan` plan."""
        unpacked = []
        for m, p in zip(self._movables, arm.parameters.values(), strict=True):
            unpacked.append(m)
            unpacked.append([p])
        return unpacked

    def run(self, trial: Trial, **kwargs):
        # TODO: Can probably do a yield from here instead and move the RunEngine call
        # to the outermost part of execution.
        # RE(trial.run()) or something like that.
        uid = self._RE(list_scan(self._readables, *self._unpack_arm(trial.arm)))
        return {"uid": uid}


class BlopMetric(Metric):
    def __init__(self, param_names: Sequence[str | HasName], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_names: frozenset[str] = frozenset(p.name if isinstance(p, HasName) else p for p in param_names)

    @property
    def param_names(self) -> frozenset[str]:
        return self._param_names


class TiledMetric(BlopMetric):
    def __init__(self, tiled_client: Container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiled_client = tiled_client

    def fetch_trial_data(self, trial: Trial, **kwargs):
        # TODO: Call tiled to get the data back
        ...


class DatabrokerMetric(BlopMetric):
    def __init__(self, broker: Broker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._broker = broker

    @abstractmethod
    def compute(self, *args, **kwargs) -> float: ...

    def fetch_trial_data(self, trial: Trial, **kwargs) -> Data:
        records = []
        uid = trial.run_metadata["uid"]

        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": self.compute(**params),
                    "sem": 0.0,
                }
            )

        uid = trial.run_metadata["uid"]
        return self._broker[uid].table(fill=True)
