import logging
from collections import defaultdict
from collections.abc import Callable, Generator

import pandas as pd
from ax import Client
from ax.analysis import Analysis, AnalysisCard, ContourPlot
from ax.api.types import TOutcome, TParameterization, TParameterValue
from bluesky.plans import list_scan
from bluesky.protocols import Movable, Readable
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
        parameters = configure_parameters(self.dofs.values())
        objectives, objective_constraints = configure_objectives(self.objectives.values())
        metrics = configure_metrics(self.objectives.values())

        self.client.configure_experiment(
            parameters, name=name, description=description, experiment_type=experiment_type, owner=owner
        )
        self.client.configure_optimization(objectives, objective_constraints)
        self.client.configure_metrics(metrics)

    def ask(self, n: int = 1) -> dict[int, TParameterization]:
        return self.client.get_next_trials(n)

    def tell(self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None) -> None:
        for trial_index in trials.keys():
            self.client.complete_trial(
                trial_index=trial_index, raw_data=outcomes[trial_index] if outcomes is not None else None
            )

    def attach_data(self, data: list[tuple[TParameterization, TOutcome]]) -> None:
        for parameters, raw_data in data:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=raw_data, progression=0)

    def learn(self, iterations: int = 1, n: int = 1) -> Generator[dict[int, TOutcome], None, None]:
        for _ in range(iterations):
            trials = self.ask(n)
            data = yield from self.acquire(trials)
            self.tell(trials, data)

    def _unpack_parameters(self, parameterizations: list[TParameterization]) -> list[Movable | TParameterValue]:
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

    def acquire(
        self, trials: dict[int, TParameterization]
    ) -> Generator[dict[int, TOutcome], None, dict[int, TOutcome] | None]:
        plan_args = self._unpack_parameters(trials.values())
        uid = yield from list_scan(self.readables, *plan_args, md={"ax_trial_indices": list(trials.keys())})
        results_df = self.db[uid].table(fill=True)
        active_objectives = [objective for objective in self.objectives.values() if objective.active]
        return {
            trial_index: self.digestion(trial_index, active_objectives, results_df, **self.digestion_kwargs)
            for trial_index in trials.keys()
        }

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
