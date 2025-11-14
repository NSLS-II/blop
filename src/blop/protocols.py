from collections.abc import Sequence
from typing import Any, Mapping, Protocol

from bluesky.protocols import NamedMovable, Readable
from bluesky.utils import MsgGenerator, plan


class Agent(Protocol):

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