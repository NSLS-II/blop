import re
from collections.abc import Sequence
from dataclasses import dataclass

from ax.api.protocols import IMetric


@dataclass(frozen=True, kw_only=True)
class Objective:
    """
    An objective to optimize.

    Attributes
    ----------
    name : str
        The name of the objective.
    minimize : bool
        Whether to minimize or maximize the objective.

    Examples
    --------
    >>> from blop.ax.objective import Objective
    >>> objective = Objective(name="objective1", minimize=True)
    """

    name: str
    minimize: bool


class ScalarizedObjective:
    """
    A scalarized objective is a weighted sum of other objectives.

    Parameters
    ----------
    expression : str
        The expression to evaluate containing the objectives.
    minimize : bool
        Whether to minimize or maximize the expression.
    **objective_names : str
        Keyword arguments mapping variables in the expression to objective names.

    Examples
    --------
    >>> from blop.ax.objective import ScalarizedObjective
    >>> scalarized_objective = ScalarizedObjective(
    ...     expression="2 * x - 4 * y",
    ...     minimize=False,
    ...     x="objective1",
    ...     y="objective2",
    ... )
    >>> print(scalarized_objective)
    2 * objective1 - 4 * objective2
    """

    def __init__(self, expression: str, *, minimize: bool, **objective_names: str):
        self._expression = expression
        self._objective_names = objective_names
        self._minimize = minimize

        if not self._objective_names:
            raise ValueError("ScalarizedObjective requires at least one objective.")

        invalidated = [key for key in self._objective_names if key not in self._expression]
        if invalidated:
            raise ValueError(f"Objectives {invalidated} not found in expression '{self._expression}'.")

    @property
    def ax_expression(self) -> str:
        """Convert the scalarized objective to a string that can be used by Ax."""
        template = self._expression
        for key in self._objective_names:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        objective_str = template.format(**self._objective_names)
        return f"-({objective_str})" if self._minimize else objective_str

    def __repr__(self) -> str:
        return (
            f"ScalarizedObjective('{self._expression}', minimize={self._minimize}, "
            f"{', '.join([f'{k}={v}' for k, v in self._objective_names.items()])})"
        )

    def __str__(self) -> str:
        return self.ax_expression


class OutcomeConstraint:
    """
    A constraint on an outcome of a suggestion. This is a *soft* constraint,
    meaning that the constraint is not guaranteed to be satisfied at all times during optimization.

    Parameters
    ----------
    constraint : str
        The constraint expression to evaluate.
    **outcomes : Objective | IMetric
        Keyword arguments mapping variables in the expression to objectives or metrics.
    """

    def __init__(self, constraint: str, **outcomes: Objective | IMetric):
        self._constraint = constraint
        self._outcomes = outcomes
        self._validate_outcomes()

    def _validate_outcomes(self) -> None:
        if not self._outcomes:
            raise ValueError("OutcomeConstraint requires at least one outcome.")

        invalidated = [name for name in self._outcomes if name not in self._constraint]
        if invalidated:
            raise ValueError(f"Outcomes {invalidated} not found in constraint '{self._constraint}'.")

    @property
    def ax_constraint(self) -> str:
        """Convert the constraint to a string that can be used by Ax."""
        template = self._constraint
        for key in self._outcomes:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        return template.format(**{key: outcome.name for key, outcome in self._outcomes.items()})

    def __repr__(self) -> str:
        outcomes_str = ", ".join(f"{k}={v.name}" for k, v in self._outcomes.items())
        return f"OutcomeConstraint('{self._constraint}', {outcomes_str})"

    def __str__(self) -> str:
        return self.ax_constraint


def to_ax_objective_str(objectives: Sequence[Objective]) -> str:
    """
    Convert a list of objectives to a string that can be used by Ax.

    Parameters
    ----------
    objectives : Sequence[Objective]
        The objectives to convert to a string.

    Returns
    -------
    str
        The string representation of the objectives.

    Examples
    --------
    >>> objectives = [Objective(name="objective1", minimize=True), Objective(name="objective2", minimize=False)]
    >>> to_ax_objective_str(objectives)
    -objective1, objective2
    """
    return ", ".join([o.name if not o.minimize else f"-{o.name}" for o in objectives])
