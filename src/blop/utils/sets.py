import warnings
from typing import cast

import numpy as np

warnings.warn("The sets module is deprecated and will be removed in Blop v1.0.0.", DeprecationWarning, stacklevel=2)


def validate_set(s, type="continuous") -> set | tuple[float, float]:
    """
    Check
    """
    if type == "continuous":
        s = cast(tuple[float, float], s)
        if len(s) == 2:
            try:
                x1, x2 = float(s[0]), float(s[1])
                if x1 <= x2:
                    return cast(tuple[float, float], (x1, x2))
            except Exception:
                pass
        raise ValueError(
            f"Invalid continuous set {s}; valid continuous sets it must be a tuple of two numbers x1, x2 such that x2 >= x1"
        )
    else:
        return cast(set, s)


def element_of(x, s, type: str = "continuous") -> bool:
    """
    Check if x is an element of s.
    """
    valid_set = validate_set(s, type=type)
    if type == "continuous":
        valid_set = cast(tuple[float, float], valid_set)
        return (x >= valid_set[0]) & (x <= valid_set[1])
    else:
        return np.isin(list(x), list(cast(set, valid_set)))


def is_subset(s1, s2, type: str = "continuous", proper: bool = False) -> bool:
    """
    Check if the set x1 is a subset of x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        s1 = cast(tuple[float, float], s1)
        s2 = cast(tuple[float, float], s2)
        if proper:
            if (s1[0] > s2[0]) and (s1[1] < s2[1]):
                return True
        else:
            if (s1[0] >= s2[0]) and (s1[1] <= s2[1]):
                return True
        return False
    else:
        s1 = cast(set, s1)
        s2 = cast(set, s2)
        return np.isin(list(s1), list(s2)).all()


def union(s1, s2, type: str = "continuous") -> tuple:
    """
    Compute the union of sets x1 and x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        s1 = cast(tuple[float, float], s1)
        s2 = cast(tuple[float, float], s2)
        new_min, new_max = min(s1[0], s2[0]), max(s1[1], s2[1])
        if new_min <= new_max:
            return (new_min, new_max)
        return None
    else:
        s1 = cast(set, s1)
        s2 = cast(set, s2)
        return s1 | s2


def intersection(s1, s2, type: str = "continuous") -> tuple:
    """
    Compute the intersection of sets x1 and x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        s1 = cast(tuple[float, float], s1)
        s2 = cast(tuple[float, float], s2)
        new_min, new_max = max(s1[0], s2[0]), min(s1[1], s2[1])
        if new_min <= new_max:
            return (new_min, new_max)
        return None
    else:
        s1 = cast(set, s1)
        s2 = cast(set, s2)
        return s1 & s2
