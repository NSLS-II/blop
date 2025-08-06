import logging
from collections import defaultdict
from collections.abc import Callable, Generator

import pandas as pd
from ax import Client
from ax.api.types import TOutcome, TParameterization, TParameterValue
from bluesky.plans import list_scan
from bluesky.protocols import Movable, Readable
from bluesky.utils import Msg
from databroker import Broker

from ...dofs import DOF
from ...objectives import Objective
from .adapters import configure_metrics, configure_objectives, configure_parameters

logger = logging.getLogger(__name__)


def default_digestion_function(trial_index: int, objectives: list[Objective], df: pd.DataFrame) -> TOutcome:
    """
    Simple digestion function.

    Assumes the following:
    - Objective names are the same as the names of the columns in the dataframe.
    - Each row in the dataframe corresponds to a single trial.

    Parameters
    ----------
    trial_index : int
        The index of the trial in the dataframe.
    objectives : list[Objective]
        The objectives of the experiment.
    df : pd.DataFrame
        The dataframe containing the results of the experiment.

    Returns
    -------
    TOutcome
        A dictionary mapping objective names to their mean and standard error. Since there
        is a single trial, the standard error is None.
    """
    return {objective.name: (df.loc[(trial_index % len(df)) + 1, objective.name], None) for objective in objectives}


class AxAgent:
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
    db : Broker
        The databroker instance to read back data from a Bluesky run.
    digestion : Callable[[pd.DataFrame], dict[str, tuple[float, float]]]
        The function to produce objective values from a dataframe of acquisition results.
    digestion_kwargs : dict
        Additional keyword arguments to pass to the digestion function.
    """

    def __init__(
        self,
        readables: list[Readable],
        dofs: list[DOF],
        objectives: list[Objective],
        db: Broker,
        digestion: Callable[[pd.DataFrame], dict[str, tuple[float, float]]] = default_digestion_function,
        digestion_kwargs: dict | None = None,
    ):
        self.readables = readables
        self.dofs = {dof.name: dof for dof in dofs}
        self.objectives = {obj.name: obj for obj in objectives}
        self.client = Client()
        self.digestion = digestion
        self.digestion_kwargs = digestion_kwargs or {}
        self.db = db

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
        results_df = self.db[uid].table(fill=True)
        active_objectives = [objective for objective in self.objectives.values() if objective.active]
        return {
            trial_index: self.digestion(trial_index, active_objectives, results_df, **self.digestion_kwargs)
            for trial_index in trials.keys()
        }
