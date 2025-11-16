from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from bluesky.protocols import NamedMovable, Readable
from bluesky.utils import MsgGenerator, plan


@runtime_checkable
class Generator(Protocol):
    """
    A minimal generator interface for optimization.
    """

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
            Each dictionary must contain a unique "_id" key to identify each parameterization.
        """
        ...

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        If points are from an external source, the dictionaries must contain keys for the DOF names.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        ...


@runtime_checkable
class EvaluationFunction(Protocol):
    def __call__(self, uid: str, *args: Any, **kwargs: Any) -> list[dict]:
        """
        Evaluate the data from a Bluesky run and produce outcomes.

        Parameters
        ----------
        uid: str
            The unique identifier of the Bluesky run to evaluate.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the outcomes of the run, one for each suggested parameterization.
        """
        ...


@runtime_checkable
class AcquisitionPlan(Protocol):
    @plan
    def __call__(
        self,
        movables: Mapping[NamedMovable, Sequence[Any]],
        readables: Sequence[Readable] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> MsgGenerator[str]:
        """
        Acquire data for optimization.

        This should be a Bluesky plan that moves the movables to each of their suggested positions
        and acquires data from the readables.

        Parameters
        ----------
        movables: Mapping[NamedMovable, Sequence[Any]]
            The movables to move and the inputs to move them to.
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
    generator: Generator
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

    generator: Generator
    movables: Sequence[NamedMovable]
    readables: Sequence[Readable]
    evaluation_function: EvaluationFunction
    acquisition_plan: AcquisitionPlan | None = None
