import logging
from collections import defaultdict
from collections.abc import Callable, Generator
from typing import Any, Concatenate, Literal, ParamSpec

import databroker
import pandas as pd
from ax import Client
from ax.analysis import Analysis, AnalysisCard, ContourPlot
from ax.api.types import TOutcome, TParameterization, TParameterValue
from ax.generation_strategy.generation_strategy import GenerationStrategy
from bluesky.plans import list_scan
from bluesky.protocols import Movable, Readable
from bluesky.utils import Msg
from databroker import Broker
from tiled.client.container import Container

from ..data_access import DatabrokerDataAccess, TiledDataAccess
from ..digestion_function import default_digestion_function
from ..dofs import DOF
from ..objectives import Objective
from .adapters import configure_metrics, configure_objectives, configure_parameters

logger = logging.getLogger(__name__)

P = ParamSpec("P")
DigestionFunction = Callable[Concatenate[int, dict[str, list[Any]], P], TOutcome]


class Agent:
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
    db : Broker | Container
        The databroker or tiled instance to read back data from a Bluesky run.
    digestion : DigestionFunction
        The function to produce objective values from a dataframe of acquisition results.
    digestion_kwargs : dict
        Additional keyword arguments to pass to the digestion function.
    """

    def __init__(
        self,
        readables: list[Readable],
        dofs: list[DOF],
        objectives: list[Objective],
        db: Broker | Container,
        digestion: DigestionFunction = default_digestion_function,
        digestion_kwargs: dict | None = None,
    ):
        self.readables = readables
        self.dofs = {dof.name: dof for dof in dofs}
        self.objectives = {obj.name: obj for obj in objectives}
        self.client = Client()
        self.digestion = digestion
        self.digestion_kwargs = digestion_kwargs or {}

        if isinstance(db, Container):
            self.data_access = TiledDataAccess(db)
        elif isinstance(db, databroker.Broker):
            self.data_access = DatabrokerDataAccess(db)
        else:
            raise ValueError("Cannot run acquistion without databroker or tiled instance!")

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
            parameters, name=name, description=description, experiment_type=experiment_type, owner=owner
        )
        self.client.configure_optimization(objectives, objective_constraints)
        self.client.configure_metrics(metrics)

        # If the digestion function is the default, we need to pass the active objectives to the digestion function
        if self.digestion == default_digestion_function:
            self.digestion_kwargs["active_objectives"] = [o for o in self.objectives.values() if o.active]

    def ask(self, n: int = 1) -> dict[int, TParameterization]:
        """
        Get the next trial(s) to run.

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
        return self.client.get_next_trials(n)

    def tell(self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None) -> None:
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
                trial_index=trial_index, raw_data=outcomes[trial_index] if outcomes is not None else None
            )

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
        for parameters, raw_data in data:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=raw_data, progression=0)

    def learn(self, iterations: int = 1, n: int = 1) -> Generator[dict[int, TOutcome], None, None]:
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
        for _ in range(iterations):
            trials = self.ask(n)
            data = yield from self.acquire(trials)
            self.tell(trials, data)

    def _unpack_parameters(self, parameterizations: list[TParameterization]) -> list[Movable | TParameterValue]:
        """Unpack the parameterizations into Bluesky plan arguments."""
        unpacked_dict = defaultdict(list)
        for parameterization in parameterizations:
            for dof_name in self.dofs.keys():
                if dof_name in parameterization:
                    unpacked_dict[dof_name].append(parameterization[dof_name])
                else:
                    raise ValueError(
                        f"Parameter {dof_name} not found in parameterization. Parameterization: {parameterization}"
                    )

        unpacked_list = []
        for dof_name, values in unpacked_dict.items():
            unpacked_list.append(self.dofs[dof_name].device)
            unpacked_list.append(values)

        return unpacked_list

    def acquire(self, trials: dict[int, TParameterization]) -> Generator[Msg, str, dict[int, TOutcome] | None]:
        """
        Acquire data given a set of trials. Deploys the trials in a single Bluesky run and
        returns the outcomes of the trials computed by the digestion function.

        Parameters
        ----------
        trials : dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.

        Returns
        -------
        Generator[Msg, str, dict[int, TOutcome] | None]
            A generator that yields the outcomes of the trials.

        See Also
        --------
        bluesky.plans.list_scan : The Bluesky plan to acquire data.
        bluesky.utils.Msg : The Bluesky message type.
        """
        plan_args = self._unpack_parameters(trials.values())
        uid = yield from list_scan(self.readables, *plan_args, md={"ax_trial_indices": list(trials.keys())})
        results = self.data_access.get_data(uid)
        return {trial_index: self.digestion(trial_index, results, **self.digestion_kwargs) for trial_index in trials.keys()}

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
