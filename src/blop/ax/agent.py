import logging
import warnings
from collections.abc import Sequence
from typing import Any

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


class Agent:
    """
    An interface that uses Ax as the backend for optimization and experiment tracking.

    The Agent is the main entry point for setting up and running Bayesian optimization
    using Blop. It coordinates the DOFs, objectives, evaluation function, and optimizer
    to perform intelligent exploration of the parameter space.

    Parameters
    ----------
    readables : Sequence[Readable]
        The readables to use for acquisition. These should be the minimal set
        of readables that are needed to compute the objectives.
    dofs : Sequence[DOF]
        The degrees of freedom that the agent can control, which determine the search space.
    objectives : Sequence[Objective]
        The objectives which the agent will try to optimize.
    evaluation : EvaluationFunction
        The function to evaluate acquired data and produce outcomes.
    acquisition_plan : AcquisitionPlan | None, optional
        The acquisition plan to use for acquiring data from the beamline. If not provided,
        :func:`blop.plans.default_acquire` will be used.
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    Notes
    -----
    For more complex setups, you can configure the Ax client directly via ``self.ax_client``.

    For complete working examples of creating and using an Agent, see the tutorial
    documentation, particularly :doc:`/tutorials/simple-experiment`.

    See Also
    --------
    blop.ax.dof.RangeDOF : For continuous parameters.
    blop.ax.dof.ChoiceDOF : For discrete parameters.
    blop.ax.objective.Objective : For defining objectives.
    blop.ax.optimizer.AxOptimizer : The optimizer used internally.
    blop.plans.optimize : Bluesky plan for running optimization.
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

        Creates an immutable :class:`blop.protocols.OptimizationProblem` that
        encapsulates all components needed for optimization. This is typically
        used internally by optimization plans.

        Returns
        -------
        OptimizationProblem
            An immutable optimization problem that can be deployed via Bluesky.

        See Also
        --------
        blop.protocols.OptimizationProblem : The optimization problem dataclass.
        blop.plans.optimize : Uses the optimization problem to run optimization.
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
        Get the next point(s) to evaluate in the search space.

        Uses the Bayesian optimization algorithm to suggest promising points based
        on all previously acquired data. Each suggestion includes an "_id" key for
        tracking.

        Parameters
        ----------
        num_points : int, optional
            The number of points to suggest. Default is 1. Higher values enable
            batch optimization but may reduce optimization efficiency per iteration.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to
            evaluate next. Each dictionary includes an "_id" key for identification.
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
        Ingest evaluation results into the optimizer.

        Updates the optimizer's model with new data. Can ingest both suggested points
        (with "_id" key) and external data (without "_id" key).

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing outcomes for a trial. For suggested
            points, include the "_id" key. For external data, include DOF names and
            objective values, and omit "_id".

        Notes
        -----
        This method is typically called automatically by :meth:`optimize`. Manual usage
        is only needed for custom workflows or when ingesting external data.

        For complete examples, see :doc:`/how-to-guides/attach-data-to-experiments`.
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
            Use ``optimize`` instead.

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
        Acquire a baseline reading for reference.

        Acquires data at a specific parameterization (or current positions) to establish
        a baseline for comparison. Useful for relative outcome constraints.

        Parameters
        ----------
        parameterization : dict[str, Any] | None, optional
            The DOF values to move to before acquiring baseline. If None, acquires
            at current positions.

        Yields
        ------
        Msg
            Bluesky messages for the run engine.

        See Also
        --------
        blop.plans.acquire_baseline : The underlying Bluesky plan.
        """
        yield from acquire_baseline(self.to_optimization_problem(), parameterization=parameterization)

    def optimize(self, iterations: int = 1, n_points: int = 1) -> MsgGenerator[None]:
        """
        Run Bayesian optimization.

        Performs iterative optimization by suggesting points, acquiring data, evaluating
        outcomes, and updating the model. This is the main method for running optimization
        with an agent.

        Parameters
        ----------
        iterations : int, optional
            The number of optimization iterations to run. Default is 1. Each iteration
            suggests, evaluates, and learns from n_points.
        n_points : int, optional
            The number of points to evaluate per iteration. Default is 1. Higher values
            enable batch optimization but may reduce optimization efficiency per iteration.

        Yields
        ------
        Msg
            Bluesky messages for the run engine.

        Notes
        -----
        This is the primary method for running optimization. It handles the full loop
        of suggesting points, acquiring data, evaluating outcomes, and updating the model.

        For complete examples, see :doc:`/tutorials/simple-experiment`.

        See Also
        --------
        blop.plans.optimize : The underlying Bluesky optimization plan.
        suggest : Get point suggestions without running acquisition.
        ingest : Manually ingest evaluation results.
        """
        yield from optimize(self.to_optimization_problem(), iterations=iterations, n_points=n_points)

    def plot_objective(
        self, x_dof_name: str, y_dof_name: str, objective_name: str, *args: Any, **kwargs: Any
    ) -> list[AnalysisCard]:
        """
        Plot the predicted objective as a function of two DOFs.

        Creates a contour plot showing the model's prediction of an objective across
        the space defined by two DOFs. Useful for visualizing the optimization landscape.

        Parameters
        ----------
        x_dof_name : str
            The name of the DOF to plot on the x-axis.
        y_dof_name : str
            The name of the DOF to plot on the y-axis.
        objective_name : str
            The name of the objective to plot.
        *args : Any
            Additional positional arguments passed to Ax's compute_analyses.
        **kwargs : Any
            Additional keyword arguments passed to Ax's compute_analyses.

        Returns
        -------
        list[AnalysisCard]
            The computed analysis cards containing the plot data.

        See Also
        --------
        ax.analysis.ContourPlot : Pre-built analysis for plotting objectives.
        ax.analysis.AnalysisCard : Contains the raw and computed data.
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
