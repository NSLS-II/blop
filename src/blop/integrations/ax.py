from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, cast

import pandas as pd
from ax import Arm, Experiment, Runner, Trial
from ax.core.base_trial import BaseTrial
from ax.utils.common.result import Ok, Result
from ax.core.map_metric import MapMetric
from ax.core.map_data import MapData
from bluesky import RunEngine
from bluesky.plans import list_scan
from bluesky.protocols import HasName, Movable, NamedMovable, Readable
from databroker import Broker
from tiled.client.container import Container
from numpy.typing import NDArray


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

    def clone(self) -> "BlopRunner":
        """Create a copy of this Runner."""
        return BlopRunner(RE=self._RE, readables=self._readables, movables=self._movables)


class BlopMetric(MapMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def unpack_trial(self, trial: BaseTrial) -> pd.DataFrame:
        """Unpacks the trial data into a DataFrame where each row is the result of a single evaluation of an arm."""
        ...

    def fetch_trial_data(self, trial: BaseTrial, **kwargs) -> Result[MapData, Exception]:
        # Unpack the trial data into a dataframe where each row is
        # the result of a single evaluation of an arm.
        df = self.unpack_trial(trial)

        # Create a dataframe that includes the arm name, metric name, and trial index
        df["arm_name"] = [arm_name for arm_name in trial.arms_by_name.keys()]
        df["metric_name"] = self.name
        df["trial_index"] = trial.index

        return Ok(value=MapData(df=df))


class TiledMetric(BlopMetric):
    def __init__(self, tiled_client: Container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiled_client = tiled_client
        # Need to save these so we can clone the metric easily
        self._args = args
        self._kwargs = kwargs

    def unpack_trial(self, trial: BaseTrial) -> list[Any]:
        # TODO: Implement this
        # uid = trial.run_metadata["uid"]
        raise NotImplementedError("TiledMetric is not implemented yet.")

    def clone(self) -> "TiledMetric":
        return self.__class__(self._tiled_client, *self._args, **self._kwargs)


class DatabrokerMetric(BlopMetric):
    def __init__(self, broker: Broker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._broker = broker
        # Need to save these so we can clone the metric easily
        self._args = args
        self._kwargs = kwargs

    def unpack_trial(self, trial: BaseTrial) -> pd.DataFrame:
        """Unpacks the trial using the databroker client.

        Parameters
        ----------
        trial: BaseTrial
            The trial to unpack.

        Returns
        -------
        pd.DataFrame
            The trial data.
        """
        uid = trial.run_metadata["uid"]
        # TODO: Why is [0] needed here?
        df: pd.DataFrame = self._broker[uid][0].table(fill=True)
        return df

    def clone(self) -> "DatabrokerMetric":
        return self.__class__(self._broker, *self._args, **self._kwargs)
