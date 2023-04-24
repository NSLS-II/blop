import numpy as np
import scipy as sp
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

# these return the values of the acquisition objectives


def expected_improvement(evaluator, classifier, X):
    """
    Given a botorch fitness model "evaluator" and a botorch validation model "classifier", compute the
    expected improvement at parameters X.
    """

    model_inputs = evaluator.prepare_inputs(X.reshape(-1, X.shape[-1]))

    ei = np.exp(
        LogExpectedImprovement(evaluator.model, best_f=evaluator.train_targets.max())(model_inputs.unsqueeze(1))
        .detach()
        .numpy()
    )
    p_good = classifier.p(X)

    return ei.reshape(X.shape[:-1]) * p_good.reshape(X.shape[:-1])


def expected_gibbon(evaluator, classifier, X, n_candidates=1024):
    """
    Given a botorch fitness model "evaluator" and a botorch validation model "classifier", compute the
    expected GIBBON at parameters X (https://www.jmlr.org/papers/volume22/21-0120/21-0120.pdf)
    """
    model_inputs = evaluator.prepare_inputs(X.reshape(-1, X.shape[-1]))

    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    candidate_set = torch.as_tensor(sampler.random(n=n_candidates)).double()

    gibbon = qLowerBoundMaxValueEntropy(evaluator.model, candidate_set)(model_inputs.unsqueeze(1)).detach().numpy()
    p_good = classifier.p(X)

    return gibbon.reshape(X.shape[:-1]) * p_good.reshape(X.shape[:-1])


# these return params that maximize the objective


def max_expected_improvement(evaluator, classifier, n_test=1024):
    """
    Compute the expected improvement over quasi-random sampled parameters, and return the location of the maximum.
    """
    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_improvement(evaluator, classifier, test_X).argmax())


def max_expected_gibbon(evaluator, classifier, n_test=1024):
    """
    Compute the expected GIBBON over quasi-random sampled parameters, and return the location of the maximum.
    """
    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_gibbon(evaluator, classifier, test_X).argmax())
