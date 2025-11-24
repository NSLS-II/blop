from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from bluesky.protocols import NamedMovable, Readable
from bluesky.utils import MsgGenerator, plan

ID_KEY: Literal["_id"] = "_id"


@runtime_checkable
class Optimizer(Protocol):
    """
    A minimal optimizer interface for optimization.
    """

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Returns a set of points in the input space, to be evaulated next.

        The "_id" key is optional and can be used to identify suggested trials for later evaluation
        and ingestion.

        Parameters
        ----------
        num_points : int | None, optional
            The number of points to suggest. If not provided, will default to 1.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to evaluate next.
            Each dictionary must contain a unique "_id" key to identify each parameterization.
        """
        ...

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        The "_id" key is optional and can be used to identify points from previously suggested trials or to identify
        the point as a "baseline" trial.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        ...


@runtime_checkable
class EvaluationFunction(Protocol):
    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        """
        Evaluate the data from a Bluesky run and produce outcomes.

        Parameters
        ----------
        uid: str
            The unique identifier of the Bluesky run to evaluate.
        suggestions: list[dict]
            A list of dictionaries, each containing the parameterization of a point to evaluate.
            The "_id" key is optional and can be used to identify each suggestion.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the outcomes of the run, one for each suggested parameterization.
            The "_id" key is optional and can be used to identify each outcome.
        """
        ...


@runtime_checkable
class AcquisitionPlan(Protocol):
    @plan
    def __call__(
        self,
        suggestions: list[dict],
        movables: Sequence[NamedMovable],
        readables: Sequence[Readable] | None = None,
    ) -> MsgGenerator[str]:
        """
        Acquire data for optimization.

        This should be a Bluesky plan that moves the movables to each of their suggested positions
        and acquires data from the readables.

        Parameters
        ----------
        suggestions: list[dict]
            A list of dictionaries, each containing the parameterization of a point to evaluate.
            The "_id" key is optional and can be used to identify each suggestion. It is suggested
            to add "_id" values to the run metadata for later identification of the acquired data.
        movables: Sequence[NamedMovable]
            The movables to move to their suggested positions.
        readables: Sequence[Readable], optional
            The readables that produce data to evaluate.

        Returns
        -------
        str
            The unique identifier of the Bluesky run.
        """
        ...


@dataclass(frozen=True)
class OptimizationProblem:
    """
    An optimization problem to solve. Immutable once initialized.

    Attributes
    ----------
    optimizer: Optimizer
        Suggests points to evaluate and ingests outcomes to inform the optimization.
    movables: Sequence[NamedMovable]
        Objects that can be moved to control the beamline using the Bluesky RunEngine.
        A subset of the movables' names must match the names of suggested parameterizations.
    readables: Sequence[Readable]
        Objects that can be read to acquire data from the beamline using the Bluesky RunEngine.
    evaluation_function: EvaluationFunction
        A callable to evaluate data from a Bluesky run and produce outcomes.
    acquisition_plan: AcquisitionPlan, optional
        A Bluesky plan to acquire data from the beamline. If not provided, a default plan will be used.
    """

    optimizer: Optimizer
    movables: Sequence[NamedMovable]
    readables: Sequence[Readable]
    evaluation_function: EvaluationFunction
    acquisition_plan: AcquisitionPlan | None = None
