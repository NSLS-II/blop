import logging
import warnings
from collections.abc import Sequence
from typing import Any, ParamSpec

from ax import Client
from ax.analysis import AnalysisCard, ContourPlot
from ax.api.types import TOutcome, TParameterization
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator

from ..plans import acquire_baseline, optimize
from ..protocols import AcquisitionPlan, EvaluationFunction, OptimizationProblem
from .dof import DOF, DOFConstraint
from .objective import Objective, OutcomeConstraint, to_ax_objective_str
from .optimizer import AxOptimizer

logger = logging.getLogger(__name__)

P = ParamSpec("P")


class Agent:
    """
    An interface that uses Ax as the backend for optimization and experiment tracking.

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
    acquisition_plan : AcquisitionPlan | None, optional
        The acquisition plan to use for acquiring data from the beamline. If not provided, a default plan will be used.
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    **kwargs: Any
        Additional keyword arguments to configure the Ax experiment.
    """

    def __init__(
        self,
        readables: Sequence[Readable],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        **kwargs: Any,
    ):
        self._readables = readables
        self._dofs = {dof.parameter_name: dof for dof in dofs}
        self._objectives = {obj.name: obj for obj in objectives}
        self._evaluation_function = evaluation
        self._acquisition_plan = acquisition_plan
        self._dof_constraints = dof_constraints
        self._outcome_constraints = outcome_constraints
        self._optimizer = AxOptimizer(
            parameters=[dof.to_ax_parameter_config() for dof in dofs],
            objective=to_ax_objective_str(objectives),
            parameter_constraints=[constraint.ax_constraint for constraint in self._dof_constraints]
            if self._dof_constraints
            else None,
            outcome_constraints=[constraint.ax_constraint for constraint in self._outcome_constraints]
            if self._outcome_constraints
            else None,
            **kwargs,
        )

    @property
    def readables(self) -> Sequence[Readable]:
        return self._readables

    @property
    def dofs(self) -> Sequence[DOF]:
        return list(self._dofs.values())

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
    def dof_constraints(self) -> Sequence[DOFConstraint] | None:
        return self._dof_constraints

    @property
    def outcome_constraints(self) -> Sequence[OutcomeConstraint] | None:
        return self._outcome_constraints

    @property
    def ax_client(self) -> Client:
        return self._optimizer.ax_client

    def to_optimization_problem(self) -> OptimizationProblem:
        """
        Construct an optimization problem from the agent.

        Returns
        -------
        OptimizationProblem
            An immutable optimization problem that can be deployed via Bluesky.
        """
        return OptimizationProblem(
            optimizer=self._optimizer,
            movables=[dof.movable for dof in self.dofs if dof.movable is not None],
            readables=self.readables,
            evaluation_function=self.evaluation_function,
            acquisition_plan=self.acquisition_plan,
        )

    def suggest(self, num_points: int = 1) -> list[dict]:
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
        return self._optimizer.suggest(num_points)

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
        return self.ax_client.get_next_trials(n)

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
            self.ax_client.complete_trial(
                trial_index=trial_index, raw_data=outcomes[trial_index] if outcomes is not None else None, **kwargs
            )

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
        self._optimizer.ingest(points)

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
            "learn is deprecated. Use 'optimize' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.optimize(iterations=iterations, n_points=n)

    def acquire_baseline(self, parameterization: dict[str, Any] | None = None) -> MsgGenerator[None]:
        """
        Acquire a baseline reading. Useful for relative outcome constraints.

        Parameters
        ----------
        parameterization : dict[str, Any] | None = None
            Move the DOFs to the given parameterization, if provided.
        """
        yield from acquire_baseline(self.to_optimization_problem(), parameterization=parameterization)

    def optimize(self, iterations: int = 1, n_points: int = 1) -> MsgGenerator[None]:
        """
        Optimize using the configured agent.

        Parameters
        ----------
        iterations : int, optional
            The number of optimization iterations to run.
        n_points : int, optional
            The number of trials to run per iteration. Higher values can lead to more efficient data acquisition,
            but slower optimization progress.
        """
        yield from optimize(self.to_optimization_problem(), iterations=iterations, n_points=n_points)

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
        return self.ax_client.compute_analyses(
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
