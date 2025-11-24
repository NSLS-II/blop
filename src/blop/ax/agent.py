import logging
import warnings
from collections.abc import Sequence
from typing import Any, ParamSpec

from ax import Client
from ax.analysis import AnalysisCard, ContourPlot
from ax.api.types import TOutcome, TParameterization
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator

from ..dofs import DOF, DOFConstraint
from ..objectives import Objective
from ..plans import optimize
from ..protocols import AcquisitionPlan, EvaluationFunction, OptimizationProblem, Optimizer
from .adapters import configure_metrics, configure_objectives, configure_parameters

logger = logging.getLogger(__name__)

P = ParamSpec("P")


class Agent(Optimizer):
    """
    An optimizer that uses Ax as the backend for optimization and experiment tracking.

    For more complex setups, you can configure the Ax client directly at ``self.ax_client``.

    Attributes
    ----------
    readables : list[Readable]
        The readables to use for acquisition. These should be the minimal set
        of readables that are needed to compute the objectives.
    dofs : list[DOF]
        The degrees of freedom that the agent can control, which determine the output of the model.
    objectives : list[Objective]
        The objectives which the agent will try to optimize.
    evaluation_function : EvaluationFunction
        The function to evaluate acquired data and produce outcomes.
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    acquisition_plan : AcquisitionPlan | None, optional
        The acquisition plan to use for acquiring data from the beamline. If not provided, a default plan will be used.
    **kwargs: Any
        Additional keyword arguments to configure the Ax experiment.
    """

    def __init__(
        self,
        readables: Sequence[Readable],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation: EvaluationFunction,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        acquisition_plan: AcquisitionPlan | None = None,
        **kwargs: Any,
    ):
        self._readables = readables
        self._dofs = {dof.name: dof for dof in dofs}
        self._dof_constraints = dof_constraints
        self._objectives = {obj.name: obj for obj in objectives}
        self._evaluation_function = evaluation
        self._acquisition_plan = acquisition_plan
        self._client = Client()
        self._configure_experiment(**kwargs)

    @property
    def readables(self) -> Sequence[Readable]:
        return self._readables

    @property
    def dofs(self) -> Sequence[DOF]:
        return list(self._dofs.values())

    @property
    def dof_constraints(self) -> Sequence[DOFConstraint] | None:
        return self._dof_constraints

    @property
    def objectives(self) -> Sequence[Objective]:
        return list(self._objectives.values())

    @property
    def evaluation_function(self) -> EvaluationFunction:
        return self._evaluation_function

    @property
    def acquisition_plan(self) -> AcquisitionPlan | None:
        return self._acquisition_plan

    @property
    def ax_client(self) -> Client:
        return self._client

    def _configure_experiment(
        self,
        name: str | None = None,
        description: str | None = None,
        experiment_type: str | None = None,
        owner: str | None = None,
    ) -> None:
        """
        Configure an experiment. Uses the active DOFs and objectives.

        Parameters
        ----------
        name : str, optional
            A name for the experiment.
        description : str, optional
            A description for the experiment.
        experiment_type : str, optional
            The type of experiment.
        owner : str, optional
            The owner of the experiment.

        See Also
        --------
        ax.Client.configure_experiment : The Ax method to configure an experiment.
        ax.Client.configure_optimization : The Ax method to configure the optimization.
        ax.Client.configure_metrics : The Ax method to configure tracking metrics.
        """
        parameters = configure_parameters(self.dofs)
        objectives, objective_constraints = configure_objectives(self.objectives)
        # TODO: Configure metrics separately, there should not be a concept of an "inactive" objective.
        metrics = configure_metrics(self.objectives)

        self._client.configure_experiment(
            parameters,
            parameter_constraints=[constraint.to_ax_constraint() for constraint in self.dof_constraints]
            if self.dof_constraints
            else None,
            name=name,
            description=description,
            experiment_type=experiment_type,
            owner=owner,
        )
        self._client.configure_optimization(objectives, objective_constraints)
        self._client.configure_metrics(metrics)

    def to_optimization_problem(self) -> OptimizationProblem:
        """
        Construct an optimization problem from the agent.

        Returns
        -------
        OptimizationProblem
            An immutable optimization problem that can be deployed via Bluesky.
        """
        return OptimizationProblem(
            optimizer=self,
            movables=[dof.movable for dof in self.dofs],
            readables=self.readables,
            evaluation_function=self.evaluation_function,
            acquisition_plan=self.acquisition_plan,
        )

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

    def ask(self, n: int = 1) -> dict[int, TParameterization]:
        """
        Get the next trial(s) to run.

        .. deprecated:: v0.8.1
            Use suggest instead.

        Parameters
        ----------
        n : int, optional
            The number of trials to get. Higher values can lead to more efficient data acquisition,
            but slower optimization progress.

        Returns
        -------
        dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        """
        warnings.warn("ask is deprecated. Use suggest instead.", DeprecationWarning, stacklevel=2)
        return self._client.get_next_trials(n)

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
            outcomes = {k: v for k, v in point.items() if k in self._objectives.keys()}
            trial_id = point.pop("_id", None)
            if trial_id is None:
                parameters = {k: v for k, v in point.items() if k in self._dofs.keys()}
                self._attach_single_trial(parameters=parameters, outcomes=outcomes)
            elif trial_id == "baseline":
                parameters = {k: v for k, v in point.items() if k in self._dofs.keys()}
                trial_index = self._client.attach_baseline(parameters=parameters)
                self._client.complete_trial(trial_index=trial_index, raw_data=outcomes)
            else:
                self._client.complete_trial(trial_index=trial_id, raw_data=outcomes)

    def _complete_trials(
        self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None, **kwargs: Any
    ) -> None:
        """
        Complete trial(s) by providing the outcomes.

        Parameters
        ----------
        trials : dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        outcomes : dict[int, TOutcome], optional
            A dictionary mapping trial indices to their outcomes. If not provided, the trial will be completed
            with no outcomes.

        See Also
        --------
        ax.Client.complete_trial : The Ax method to complete a trial.
        """
        for trial_index in trials.keys():
            self._client.complete_trial(
                trial_index=trial_index, raw_data=outcomes[trial_index] if outcomes is not None else None, **kwargs
            )

    def tell(self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None) -> None:
        """
        Complete trial(s) by providing the outcomes.

        .. deprecated:: v0.8.1
            Use ingest instead.

        Parameters
        ----------
        trials : dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        outcomes : dict[int, TOutcome], optional
            A dictionary mapping trial indices to their outcomes. If not provided, the trial will be completed
            with no outcomes.

        See Also
        --------
        ax.Client.complete_trial : The Ax method to complete a trial.
        """
        warnings.warn("tell is deprecated. Use ingest instead.", DeprecationWarning, stacklevel=2)
        return self._complete_trials(trials=trials, outcomes=outcomes)

    def _attach_single_trial(self, parameters: TParameterization, outcomes: TOutcome) -> None:
        """Attach a single trial to the experiment."""
        trial_index = self._client.attach_trial(parameters=parameters)
        self._client.complete_trial(trial_index=trial_index, raw_data=outcomes, progression=0)

    def learn(self, iterations: int = 1, n: int = 1) -> MsgGenerator[None]:
        """
        Learn by running trials and providing the outcomes.

        .. deprecated:: v0.9.0
            Use blop.plans.optimize with self.to_optimization_problem instead.

        Parameters
        ----------
        iterations : int, optional
            The number of optimization iterations to run.
        n : int, optional
            The number of trials to run per iteration. Higher values can lead to more efficient data acquisition,
            but slower optimization progress.
        """
        warnings.warn(
            "learn is deprecated. Use blop.plans.optimize with self.to_optimization_problem instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from optimize(self.to_optimization_problem(), iterations=iterations, n_points=n)

    def plot_objective(
        self, x_dof_name: str, y_dof_name: str, objective_name: str, *args: Any, **kwargs: Any
    ) -> list[AnalysisCard]:
        """
        Plot the predicted objective as a function of the two DOFs.

        Parameters
        ----------
        x_dof_name : str
            The name of the DOF to plot on the x-axis.
        y_dof_name : str
            The name of the DOF to plot on the y-axis.
        objective_name : str
            The name of the objective to plot.

        Returns
        -------
        list[AnalysisCard]
            The computed analysis cards

        See Also
        --------
        ax.analysis.ContourPlot : Pre-built analysis for plotting the objective as a function of two parameters.
        ax.analysis.AnalysisCard : The Ax analysis card class which contains the raw and computed data.
        """
        return self._client.compute_analyses(
            [
                ContourPlot(
                    x_parameter_name=x_dof_name,
                    y_parameter_name=y_dof_name,
                    metric_name=objective_name,
                ),
            ],
            *args,
            **kwargs,
        )
