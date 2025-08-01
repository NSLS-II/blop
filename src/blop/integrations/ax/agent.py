import logging
from collections import defaultdict
from collections.abc import Callable, Generator
from typing import Literal

import pandas as pd
from ax import Client
from ax.api.types import TOutcome, TParameterization, TParameterValue
from ax.generation_strategy.generation_strategy import GenerationStrategy
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
        The index of the trial.
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
            `this tutorial<https://ax.dev/docs/tutorials/modular_botorch/>`_ for more details.

        See Also
        --------
        configure_generation_strategy : Configure an implicit generation strategy for the experiment.
        """
        self.client.set_generation_strategy(generation_strategy)

    def summarize(self) -> pd.DataFrame:
        return self.client.summarize()
