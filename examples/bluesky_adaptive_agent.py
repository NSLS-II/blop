from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from bluesky_adaptive.agents.base import Agent as BlueskyAdaptiveBaseAgent  # type: ignore[import-untyped]
from databroker.client import BlueskyRun  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from blop.agent import BaseAgent as BlopAgent  # type: ignore[import-untyped]
from blop.digestion import default_digestion_function  # type: ignore[import-untyped]


class BlueskyAdaptiveAgent(BlueskyAdaptiveBaseAgent, BlopAgent):
    """A BlueskyAdaptiveAgent that uses Blop for the underlying agent."""

    # TODO: Move into main package once databroker V2 is supported

    def __init__(
        self,
        *,
        acqf_string: str,
        route: bool,
        sequential: bool,
        upsample: int,
        acqf_kwargs: dict[str, Any],
        detector_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._acqf_string = acqf_string
        self._route = route
        self._sequential = sequential
        self._upsample = upsample
        self._acqf_kwargs = acqf_kwargs
        self._detector_names = detector_names or []

    @property
    def detector_names(self) -> list[str]:
        return [str(name) for name in self._detector_names]

    @detector_names.setter
    def detector_names(self, names: list[str]):
        self._detector_names = list(names)

    @property
    def acquisition_function(self) -> str:
        return str(self._acqf_string)

    @acquisition_function.setter
    def acquisition_function(self, acqf_string: str):
        self._acqf_string = str(acqf_string)

    @property
    def route(self) -> bool:
        return bool(self._route)

    @route.setter
    def route(self, route: bool):
        self._route = route

    @property
    def sequential(self) -> bool:
        return bool(self._sequential)

    @sequential.setter
    def sequential(self, sequential: bool):
        self._sequential = sequential

    @property
    def upsample(self) -> int:
        return int(self._upsample)

    @upsample.setter
    def upsample(self, upsample: int):
        self._upsample = int(upsample)

    @property
    def acqf_kwargs(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._acqf_kwargs.items()}

    def update_acqf_kwargs(self, **kwargs):
        self._acqf_kwargs.update(kwargs)

    def server_registrations(self) -> list[str]:
        """This is how we make these avaialble to the REST API."""
        self._register_method("Update Acquistion Function Kwargs", self.update_acqf_kwargs)
        self._register_property("Acquisition Function", self.acquisition_function, self.acquisition_function)
        self._register_property("Route Points", self.route, self.route)
        self._register_property("Sequential Points", self.sequential, self.sequential)
        self._register_property("Upsample Points", self.upsample, self.upsample)
        return super().server_registrations()

    def ask(self, batch_size: int) -> tuple[Sequence[dict[str, ArrayLike]], Sequence[ArrayLike]]:
        default_result = super().ask(
            n=batch_size,
            acqf=self._acqf_string,
            route=self._route,
            sequential=self._sequential,
            upsample=self._upsample,
            **self._acqf_kwargs,
        )

        """res = {
            "points": {dof.name: list(points[..., i]) for i, dof in enumerate(active_dofs(read_only=False))},
            "acqf_name": acqf_config["name"],
            "acqf_obj": list(np.atleast_1d(acqf_obj.numpy())),
            "acqf_kwargs": acqf_kwargs,
            "duration_ms": duration,
            "sequential": sequential,
            "upsample": upsample,
            "read_only_values": read_only_values,
            # "posterior": p,
        }
        """

        points: dict[str, list[ArrayLike]] = default_result.pop("points")
        acqf_obj: list[ArrayLike] = default_result.pop("acqf_obj")
        # Turn dict of list of points into list of consistently sized points
        points: list[tuple[ArrayLike]] = list(zip(*[value for _, value in points.items()], strict=False))
        dicts = []
        for point, obj in zip(points, acqf_obj, strict=False):
            d = default_result.copy()
            d["point"] = point
            d["acqf_obj"] = obj
            dicts.append(d)
        return points, dicts

    def tell(self, x: dict[str, ArrayLike], y: dict[str, ArrayLike]):
        x = {key: x_i for x_i, key in zip(x, self.dofs.names, strict=False)}
        y = {key: y_i for y_i, key in zip(y, self.objectives.names, strict=False)}
        super().tell(data={**x, **y})
        return {**x, **y}

    def report(self) -> dict[str, Any]:
        raise NotImplementedError("Report is not implmented for BlueskyAdaptiveAgent")

    def unpack_run(self, run: BlueskyRun) -> tuple[list[ArrayLike], list[ArrayLike]]:
        """Use my DOFs to convert the run into an independent array, and my objectives to create the dependent array.
        In practice for shape management, we will use lists not np.arrays at this stage.
        Parameters
        ----------
        run : BlueskyRun

        Returns
        -------
        independent_var :
            The independent variable of the measurement
        dependent_var :
            The measured data, processed for relevance
        """
        if not self.digestion or self.digestion == default_digestion_function:
            # Assume all raw data is available in primary stream as keys
            return (
                [run.primary.data[key].read() for key in self.dofs.names],
                [run.primary.data[key].read() for key in self.objectives.names],
            )
        else:
            # Hope and pray that the digestion function designed for DataFrame can handle the XArray
            data: pd.DataFrame = self.digestion(run.primary.data.read(), **self.digestion_kwargs)
            return [data.loc[:, key] for key in self.dofs.names], [data.loc[:, key] for key in self.objectives.names]

    def measurement_plan(self, point: ArrayLike) -> tuple[str, list[Any], dict[str, Any]]:
        """Fetch the string name of a registered plan, as well as the positional and keyword
        arguments to pass that plan.

        Args/Kwargs is a common place to transform relative into absolute motor coords, or
        other device specific parameters.

        By default, this measurement plan attempts to use in the built in functionality in a QueueServer compatible way.
        Signals and Devices are not passed as objects, but serialized as strings for the RE as a service to use.

        Parameters
        ----------
        point : ArrayLike
            Next point to measure using a given plan

        Returns
        -------
        plan_name : str
        plan_args : List
            List of arguments to pass to plan from a point to measure.
        plan_kwargs : dict
            Dictionary of keyword arguments to pass the plan, from a point to measure.
        """
        if isinstance(self.acquisition_plan, Callable):
            plan_name = self.acquisition_plan.__name__
        else:
            plan_name = self.acquisition_plan
        if plan_name == "default_acquisition_plan":
            # Convert point back to dict form for the sake of compatability with default plan
            acquisition_dofs = self.dofs(active=True, read_only=False)

            return self.acquisition_plan.__name__, [
                acquisition_dofs,
                {dof.name: point[i] for i, dof in enumerate(acquisition_dofs)},
                [*self.detector_names, *[dev.__name__ for dev in self.dofs.devices]],
            ]
        else:
            raise NotImplementedError("Only default_acquisition_plan is implemented")
