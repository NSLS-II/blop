from typing import Any, Sequence

from ax import (
    Client, RangeParameterConfig, ChoiceParameterConfig)

from ..protocols import Optimizer


class AxOptimizer(Optimizer):
    """
    An optimizer that uses Ax as the backend for optimization and experiment tracking.

    Parameters
    ----------
    parameters : Sequence[RangeParameterConfig | ChoiceParameterConfig]
        The parameters to optimize.
    objective : str
        The objective to optimize.
    parameter_constraints : Sequence[str] | None, optional
        The parameter constraints to apply to the optimization.
    outcome_constraints : Sequence[str] | None, optional
        The outcome constraints to apply to the optimization.
    client_kwargs: dict[str, Any] | None, optional
        Additional keyword arguments to configure the client.
    **kwargs: Any
        Additional keyword arguments to configure the experiment.
    """
    def __init__(
        self,
        parameters: Sequence[RangeParameterConfig | ChoiceParameterConfig],
        objective: str,
        parameter_constraints: Sequence[str] | None = None,
        outcome_constraints: Sequence[str] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._parameter_names = [parameter.name for parameter in parameters]
        self._client = Client(**(client_kwargs or {}))
        self._client.configure_experiment(
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            **kwargs,
        )
        self._client.configure_optimization(
            objective=objective,
            outcome_constraints=outcome_constraints,
        )

    @property
    def ax_client(self) -> Client:
        return self._client

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Returns a set of points in the input space, to be evaulated next.

        Parameters
        ----------
        num_points : int | None, optional
            The number of points to suggest. If not provided, will default to 1.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to evaluate next.
        """
        if num_points is None:
            num_points = 1
        next_trials = self._client.get_next_trials(max_trials=num_points)
        return [
            {
                "_id": trial_index,
                **parameterization,
            }
            for trial_index, parameterization in next_trials.items()
        ]

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        If points are from an external source, each dictionary must contain keys for the DOF names, the objectives, and
        the "_id" key must be omitted.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing at least the outcome(s) of a trial
            and optionally the associated parameterization
        """
        for point in points:
            trial_idx = point.pop("_id", None)
            if trial_idx is None:
                parameters = {k: v for k, v in point.items() if k in self._parameter_names}
                trial_idx = self._client.attach_trial(parameters=parameters)
            elif trial_idx == "baseline":
                parameters = {k: v for k, v in point.items() if k in self._parameter_names}
                trial_idx = self._client.attach_baseline(parameters=parameters)
            self._client.complete_trial(trial_index=trial_idx, raw_data=point)
