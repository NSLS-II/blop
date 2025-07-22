import logging
from collections import defaultdict
from collections.abc import Generator
from typing import Callable

from ax import Client
from ax.api.types import TParameterization, TOutcome, TParameterValue
from bluesky.protocols import Movable, Readable
from bluesky.plans import list_scan
from databroker import Broker

from .adapters import configure_parameters, configure_metrics, configure_objectives
from ...dofs import DOF
from ...objectives import Objective
from ...digestion import default_digestion_function

logger = logging.getLogger(__name__)


class AxAgent:
    def __init__(self, readables: list[Readable], dofs: list[DOF], objectives: list[Objective], db: Broker, digestion: Callable = default_digestion_function, digestion_kwargs: dict | None = None):
        self.readables = readables
        self.dofs = {dof.name: dof for dof in dofs}
        self.objectives = {obj.name: obj for obj in objectives}
        self.client = Client()
        self.digestion = digestion
        self.digestion_kwargs = digestion_kwargs or {}

    def configure_experiment(self, name: str | None = None, description: str | None = None, experiment_type: str | None = None, owner: str | None = None) -> None:
        parameters = configure_parameters(self.dofs.values())
        objectives, objective_constraints = configure_objectives(self.objectives.values())
        metrics = configure_metrics(self.objectives.values())

        self.client.configure_experiment(parameters, name=name, description=description, experiment_type=experiment_type, owner=owner)
        self.client.configure_optimization(objectives, objective_constraints)
        self.client.configure_metrics(metrics)

    def ask(self, n: int = 1) -> dict[int, TParameterization]:
        return self.client.get_next_trials(n)

    def tell(self, trials: dict[int, TParameterization], outcomes: dict[int, TOutcome] | None = None) -> None:
        for trial_index, parameters in trials.items():
            self.client.complete_trial(trial_index=trial_index, parameters=parameters, raw_data=outcomes[trial_index] if outcomes is not None else None)

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
                    raise ValueError(f"Parameter {dof_name} not found in parameterization. Parameterization: {parameterization}")

        unpacked_list = []
        for dof_name, values in unpacked_dict.items():
            unpacked_list.append(self.dofs[dof_name].device)
            unpacked_list.append(values)

        return unpacked_list
        
    def acquire(self, trials: dict[int, TParameterization]) -> Generator[dict[int, TOutcome], None, dict[int, TOutcome] | None]:
        plan_args = self._unpack_parameters(trials.values())

        uid = yield from list_scan(self.readables, *plan_args, md={"ax_trial_indices": list(trials.keys())})

        results_df = self.db[uid][0].table(fill=True)

        # TODO: Convert to dict[int, TOutcome] while using standard digestion function
        #return self.digestion(trials, results_df)
        return {}
        
