from .analytic import *  # noqa F401
from .monte_carlo import *  # noqa F401

ACQ_FUNC_CONFIG = {
    "quasi-random": {
        "identifiers": ["qr", "quasi-random"],
        "pretty_name": "Quasi-random",
        "description": "Sobol-sampled quasi-random points.",
        "multitask_only": False,
    },
    "expected_mean": {
        "identifiers": ["em", "expected_mean"],
        "pretty_name": "Expected mean",
        "multitask_only": False,
        "description": "The expected value at each input.",
    },
    "expected_improvement": {
        "identifiers": ["ei", "expected_improvement"],
        "pretty_name": "Expected improvement",
        "multitask_only": False,
        "description": r"The expected value of max(f(x) - \nu, 0), where \nu is the current maximum.",
    },
    "monte_carlo_expected_improvement": {
        "identifiers": ["qei", "monte_carlo_expected_improvement"],
        "pretty_name": "Monte Carlo Expected improvement",
        "multitask_only": False,
        "description": r"The expected value of max(f(x) - \nu, 0), where \nu is the current maximum.",
    },
    "noisy_expected_hypervolume_improvement": {
        "identifiers": ["nehvi", "noisy_expected_hypervolume_improvement"],
        "pretty_name": "Noisy expected hypervolume improvement",
        "multitask_only": True,
        "description": r"It's like a big box. How big is the box?",
    },
    "lower_bound_max_value_entropy": {
        "identifiers": ["lbmve", "lbmes", "gibbon", "lower_bound_max_value_entropy"],
        "pretty_name": "Lower bound max value entropy",
        "multitask_only": False,
        "description": r"Max entropy search, basically",
    },
    "probability_of_improvement": {
        "identifiers": ["pi", "probability_of_improvement"],
        "pretty_name": "Probability of improvement",
        "multitask_only": False,
        "description": "The probability that this input improves on the current maximum.",
    },
    "upper_confidence_bound": {
        "identifiers": ["ucb", "upper_confidence_bound"],
        "default_args": {"beta": 4},
        "pretty_name": "Upper confidence bound",
        "multitask_only": False,
        "description": r"The expected value, plus some multiple of the uncertainty (typically \mu + 2\sigma).",
    },
}
