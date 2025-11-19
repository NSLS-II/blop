import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, ParamSpec

import pandas as pd
from ax import Client
from ax.analysis import Analysis, AnalysisCard, ContourPlot
from ax.api.types import TOutcome, TParameterization
from ax.generation_strategy.generation_strategy import GenerationStrategy
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator

from ..dofs import DOF, DOFConstraint
from ..evaluation import default_evaluation_function
from ..objectives import Objective
from ..plans import optimize
from ..protocols import Generator, OptimizationProblem
from .adapters import configure_metrics, configure_objectives, configure_parameters

logger = logging.getLogger(__name__)

P = ParamSpec("P")
EvaluationFunction = Callable[Concatenate[int, dict[str, list[Any]], P], TOutcome]


class Agent(Generator):
    """
    An agent interface that uses Ax as the backend for optimization and experiment tracking.

    Attributes
    ----------
    readables : list[Readable]
        The readables to use for acquisition. These should be the minimal set
        of readables that are needed to compute the objectives.
    dofs : list[DOF]
        The degrees of freedom that the agent can control, which determine the output of the model.
    objectives : list[Objective]
        The objectives which the agent will try to optimize.
    dof_constraints : Sequence[DOFConstraint], optional
        Constraints on DOFs to refine the search space.
    evaluation : EvaluationFunction
        The function that evaluates the acquired data and produces outcomes.
    evaluation_kwargs : dict
        Additional keyword arguments to pass to the evaluation function.
    """

    def __init__(
        self,
        readables: Sequence[Readable],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        dof_constraints: Sequence[DOFConstraint] = None,
        evaluation_function: EvaluationFunction = default_evaluation_function,
        evaluation_kwargs: dict | None = None,
    ):
        self._readables = readables
        self._dofs = {dof.name: dof for dof in dofs}
        self._dof_constraints = dof_constraints
        self._objectives = {obj.name: obj for obj in objectives}
        self.client = Client()
        self._evaluation_function = evaluation_function
        self._evaluation_kwargs = evaluation_kwargs or {}

    @property
    def readables(self) -> Sequence[Readable]:
        return self._readables

    @property
    def dofs(self) -> dict[str, DOF]:
        return self._dofs

    @property
    def objectives(self) -> dict[str, Objective]:
        return self._objectives

    @property
    def dof_constraints(self) -> Sequence[DOFConstraint]:
        return self._dof_constraints

    @property
    def evaluation_function(self) -> EvaluationFunction:
        return self._evaluation_function

    @property
    def evaluation_kwargs(self) -> dict:
        return self._evaluation_kwargs

    def to_optimization_problem(self) -> OptimizationProblem:
        return OptimizationProblem(
            generator=self,
            movables=[dof.movable for dof in self.dofs.values()],
            readables=self.readables,
            evaluation_function=self.evaluation_function,
        )

    def configure_experiment(
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
        parameters = configure_parameters(self.dofs.values())
        objectives, objective_constraints = configure_objectives(self.objectives.values())
        metrics = configure_metrics(self.objectives.values())

        self.client.configure_experiment(
            parameters,
            parameter_constraints=(constraint.to_ax_constraint() for constraint in self.dof_constraints)
            if self.dof_constraints
            else None,
            name=name,
            description=description,
            experiment_type=experiment_type,
            owner=owner,
        )
        self.client.configure_optimization(objectives, objective_constraints)
        self.client.configure_metrics(metrics)

        # If the evaluation function is the default, we need to pass the active objectives to the evaluation function
        if self.evaluation_function == default_evaluation_function:
            self.evaluation_kwargs["active_objectives"] = [o for o in self.objectives.values() if o.active]

    def attach_baseline(self, parameterization: TParameterization, arm_name: str | None = None) -> TParameterization:
        """
        Attach a baseline to the experiment.

        Parameters
        ----------
        parameterization : TParameterization
            The parameterization of the baseline to attach.
        arm_name : str, optional
            A name for the arm to distinguish it from other arms.
        """
        trial_index = self.client.attach_baseline(parameters=parameterization, arm_name=arm_name)
        return {trial_index: parameterization}

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
        next_trials = self.get_next_trials(num_points)
        return [
            {
                "_id": trial_index,
                **parameterization,
            }
            for trial_index, parameterization in next_trials.items()
        ]

    def get_next_trials(self, max_trials: int = 1, **kwargs: Any) -> dict[int, TParameterization]:
        """
        Get the next trial(s) to run.

        Parameters
        ----------
        max_trials : int, optional
            The maximum number of trials to get. Higher values can lead to more efficient data acquisition,
            but slower optimization progress.

        Returns
        -------
        dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        """
        return self.client.get_next_trials(max_trials, **kwargs)

    def ask(self, n: int = 1) -> dict[int, TParameterization]:
        """
        Get the next trial(s) to run.

        .. deprecated:: v0.8.1
            Use suggest or get_next_trials instead.

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
        warnings.warn("ask is deprecated. Use get_next_trials or suggest instead.", DeprecationWarning, stacklevel=2)
        return self.get_next_trials(n)

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        If points are from an external source, the dictionaries must contain keys for the DOF names.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing at least the outcome(s) of a trial
            and optionally the associated parameterization
        """
        for point in points:
            outcomes = {k: v for k, v in point.items() if k in self.objectives.keys()}
            trial_index = point.pop("_id", None)
            if trial_index is None:
                parameters = {k: v for k, v in point.items() if k in self.dofs.keys()}
                self._attach_single_trial(parameters=parameters, outcomes=outcomes)
            else:
                self.client.complete_trial(trial_index=trial_index, raw_data=outcomes)

    def complete_trials(
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
            self.client.complete_trial(
                trial_index=trial_index, raw_data=outcomes[trial_index] if outcomes is not None else None, **kwargs
            )

    def tell(self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None) -> None:
        """
        Complete trial(s) by providing the outcomes.

        .. deprecated:: v0.8.1
            Use ingest or complete_trials instead.

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
        warnings.warn("tell is deprecated. Use ingest or complete_trials instead.", DeprecationWarning, stacklevel=2)
        return self.complete_trials(trials=trials, outcomes=outcomes)

    def _attach_single_trial(self, parameters: TParameterization, outcomes: TOutcome) -> None:
        """
        Attach a single trial to the experiment.
        """
        trial_index = self.client.attach_trial(parameters=parameters)
        self.client.complete_trial(trial_index=trial_index, raw_data=outcomes, progression=0)

    def attach_data(self, data: list[tuple[TParameterization, TOutcome]]) -> None:
        """
        Attach data to the experiment in the form of new trials. Useful for
        resuming an experiment from a previous state or manually adding data.

        Parameters
        ----------
        data : list[tuple[TParameterization, TOutcome]]
            A dataset of input and output examples.

        See Also
        --------
        ax.Client.attach_trial : The Ax method to attach a trial.
        ax.Client.complete_trial : The Ax method to complete a trial.
        """
        for parameters, outcomes in data:
            self._attach_single_trial(parameters=parameters, outcomes=outcomes)

    def compute_analyses(self, analyses: list[Analysis], display: bool = True) -> list[AnalysisCard]:
        """
        Compute analyses for the experiment.

        Parameters
        ----------
        analyses : list[Analysis]
            The Ax analyses to compute
        display : bool
            Show plots in an interactive environment.

        Returns
        -------
        list[AnalysisCard]
            The computed analysis cards

        See Also
        --------
        ax.analysis : The Ax analysis module which contains many pre-built analyses.
        ax.analysis.Analysis : The Ax analysis class to create custom analyses.
        ax.analysis.AnalysisCard : The Ax analysis card class which contains the raw and computed data.
        """
        return self.client.compute_analyses(analyses=analyses, display=display)

    def plot_objective(self, x_dof_name: str, y_dof_name: str, objective_name: str) -> list[AnalysisCard]:
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
        return self.client.compute_analyses(
            analyses=[
                ContourPlot(
                    x_parameter_name=x_dof_name,
                    y_parameter_name=y_dof_name,
                    metric_name=objective_name,
                )
            ],
            display=True,
        )

    def configure_generation_strategy(
        self,
        method: Literal["balanced", "fast", "random_search"] = "fast",
        initialization_budget: int | None = None,
        initialization_random_seed: int | None = None,
        initialize_with_center: bool = True,
        use_existing_trials_for_initialization: bool = True,
        min_observed_initialization_trials: int | None = None,
        allow_exceeding_initialization_budget: bool = False,
        torch_device: str | None = None,
    ) -> None:
        """
        Implicitly configure the models and algorithms used to generate new points. Based on the
        settings and the DOF configuration, an appropriate generation strategy is chosen.

        Parameters
        ----------
        method : Literal["balanced", "fast", "random_search"], optional
            Methods for generating new points.
        initialization_budget : int | None, optional
            The number of points to generate during the initial exploration phase. Can be set
            to 0 to disable initialization (when attaching pre-existing data, for example).
        initialization_random_seed : int | None, optional
            The random seed for initialization.
        initialize_with_center : bool, optional
            Use the first point to sample the center of the search space defined by the DOFs.
        use_existing_trials_for_initialization : bool, optional
            Use the pre-existing trials to build the initial models.
        min_observed_initialization_trials : int | None, optional
            The minimum number of trials to observe before building the initial models.
        allow_exceeding_initialization_budget : bool, optional
            Allow the initialization budget to be exceeded, when determined necessary.
        torch_device : str | None, optional
            The device to use for PyTorch tensors (e.g. "cuda", "cpu", etc.).

        See Also
        --------
        set_generation_strategy : Explicitly set the generation strategy for the experiment.
        ax.Client.configure_generation_strategy : The Ax method to configure the generation strategy.
        """
        self.client.configure_generation_strategy(
            method=method,
            initialization_budget=initialization_budget,
            initialization_random_seed=initialization_random_seed,
            initialize_with_center=initialize_with_center,
            use_existing_trials_for_initialization=use_existing_trials_for_initialization,
            min_observed_initialization_trials=min_observed_initialization_trials,
            allow_exceeding_initialization_budget=allow_exceeding_initialization_budget,
            torch_device=torch_device,
        )

    def set_generation_strategy(self, generation_strategy: GenerationStrategy) -> None:
        """
        Explicitly set the generation strategy for the experiment. This allows for finer-grained
        control over the models and algorithms used to generate new points.

        Familiarity with Ax and BoTorch internals is recommended prior to using this method.

        Parameters
        ----------
        generation_strategy : GenerationStrategy
            The generation strategy to use for the experiment. See
            `this tutorial <https://ax.dev/docs/tutorials/modular_botorch/>`_ for more details.

        See Also
        --------
        configure_generation_strategy : Configure an implicit generation strategy for the experiment.
        ax.Client.set_generation_strategy : The Ax method to set the generation strategy.
        """
        self.client.set_generation_strategy(generation_strategy)

    def summarize(self) -> pd.DataFrame:
        """
        View of the experiment state.

        NOTE: This method is a convenience method for inspecting the experiment state.
        It is not recommended to use this for downstream analysis.

        Returns
        -------
        pd.DataFrame
            A dataframe of the experiment state containing a parameterization per row.

        See Also
        --------
        ax.Client.summarize : The Ax method to summarize the experiment state.
        """
        return self.client.summarize()

    def learn(self, iterations: int = 1, n: int = 1) -> MsgGenerator[None]:
        """
        Learn by running trials and providing the outcomes.

        Parameters
        ----------
        iterations : int, optional
            The number of optimization iterations to run.
        n : int, optional
            The number of trials to run per iteration. Higher values can lead to more efficient data acquisition,
            but slower optimization progress.

        Returns
        -------
        Generator[dict[int, TOutcome], None, None]
            A generator that yields the outcomes of the trials.
        """
        yield from optimize(self.to_optimization_problem(), iterations=iterations, n_points=n)
