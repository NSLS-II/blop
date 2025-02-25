import numpy as np


def validate_set(s, type="continuous") -> tuple:
    """
    Check
    """
    if type == "continuous":
        if len(s) == 2:
            try:
                x1, x2 = float(s[0]), float(s[1])
                if x1 <= x2:
                    return (x1, x2)
            except Exception:
                pass
        raise ValueError(
            f"Invalid continuous set {s}; valid continuous sets it must be a tuple of two numbers x1, x2 such that x2 >= x1"
        )
    else:
        return set(s)


def element_of(x, s, type: str = "continuous") -> bool:
    """
    Check if x is an element of s.
    """
    validate_set(s, type=type)
    if type == "continuous":
        return (x >= s[0]) & (x <= s[1])
    else:
        return np.isin(list(x), list(s))


def is_subset(s1, s2, type: str = "continuous", proper: bool = False) -> bool:
    """
    Check if the set x1 is a subset of x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        if proper:
            if (s1[0] > s2[0]) and (s1[1] < s2[1]):
                return True
        else:
            if (s1[0] >= s2[0]) and (s1[1] <= s2[1]):
                return True
        return False
    else:
        return np.isin(list(s1), list(s2)).all()


def union(s1, s2, type: str = "continuous") -> tuple:
    """
    Compute the union of sets x1 and x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        new_min, new_max = min(s1[0], s2[0]), max(s1[1], s2[1])
        if new_min <= new_max:
            return (new_min, new_max)
        return None
    else:
        return s1 | s2


def intersection(s1, s2, type: str = "continuous") -> tuple:
    """
    Compute the intersection of sets x1 and x2.
    """
    validate_set(s1, type=type)
    validate_set(s2, type=type)
    if type == "continuous":
        new_min, new_max = max(s1[0], s2[0]), min(s1[1], s2[1])
        if new_min <= new_max:
            return (new_min, new_max)
        return None
    else:
        return s1 & s2
