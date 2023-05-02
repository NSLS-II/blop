import numpy as np
import scipy as sp
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

# these return the values of the acquisition objectives


def expected_sumtask_improvement(tasks, classifier, X):
    total_mu, total_var = 0, 0

    total_nu = np.c_[[task.regressor.Y[:, 0] for task in tasks]].sum(axis=0).max()

    for task in tasks:
        total_mu += task.regressor.mean(X)
        total_var += task.regressor.sigma(X) ** 2

    total_sigma = np.sqrt(total_var)

    esti = np.exp(-0.5 * np.square(total_mu - total_nu) / total_var) * total_sigma / np.sqrt(2 * np.pi) + 0.5 * (
        total_mu - total_nu
    ) * (1 + sp.special.erf((total_mu - total_nu) / (np.sqrt(2) * total_sigma)))

    p_good = classifier.p(X)

    return esti.reshape(X.shape[:-1]) * p_good.reshape(X.shape[:-1])


def expected_improvement(regressor, classifier, X):
    """
    Given a botorch fitness model "regressor" and a botorch validation model "classifier", compute the
    expected improvement at parameters X.
    """

    model_inputs = regressor.normalize_inputs(X.reshape(-1, X.shape[-1]))

    ei = np.exp(
        LogExpectedImprovement(regressor.model, best_f=regressor.train_targets.max())(model_inputs.unsqueeze(1))
        .detach()
        .numpy()
    )
    p_good = classifier.p(X)

    return ei.reshape(X.shape[:-1]) * p_good.reshape(X.shape[:-1])


def expected_gibbon(regressor, classifier, X, n_candidates=1024):
    """
    Given a botorch fitness model "regressor" and a botorch validation model "classifier", compute the
    expected GIBBON at parameters X (https://www.jmlr.org/papers/volume22/21-0120/21-0120.pdf)
    """
    model_inputs = regressor.normalize_inputs(X.reshape(-1, X.shape[-1]))

    sampler = sp.stats.qmc.Halton(d=regressor.X.shape[-1], scramble=True)
    candidate_set = torch.as_tensor(sampler.random(n=n_candidates)).double()

    gibbon = qLowerBoundMaxValueEntropy(regressor.model, candidate_set)(model_inputs.unsqueeze(1)).detach().numpy()
    p_good = classifier.p(X)

    return gibbon.reshape(X.shape[:-1]) * p_good.reshape(X.shape[:-1])


# these return params that maximize the objective


def max_expected_improvement(regressor, classifier, n_test=1024):
    """
    Compute the expected improvement over quasi-random sampled parameters, and return the location of the maximum.
    """
    sampler = sp.stats.qmc.Halton(d=regressor.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_improvement(regressor, classifier, test_X).argmax())


def max_expected_gibbon(regressor, classifier, n_test=1024):
    """
    Compute the expected GIBBON over quasi-random sampled parameters, and return the location of the maximum.
    """
    sampler = sp.stats.qmc.Halton(d=regressor.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_gibbon(regressor, classifier, test_X).argmax())
