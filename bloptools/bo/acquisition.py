import numpy as np
import scipy as sp
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

# these return the values of the acquisition objectives


def expected_improvement(evaluator, classifier, X):
    torch_X = torch.as_tensor(X.reshape(-1, X.shape[-1]))

    ei = np.exp(
        LogExpectedImprovement(evaluator.model, best_f=evaluator.Y.max())(torch_X.unsqueeze(1)).detach().numpy()
    )
    p_good = classifier.p(torch_X)

    return (ei * p_good).reshape(X.shape[:-1])


def expected_gibbon(evaluator, classifier, X, n_candidates=1024):
    torch_X = torch.as_tensor(X.reshape(-1, X.shape[-1]))

    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    candidate_set = torch.as_tensor(sampler.random(n=n_candidates)).double()

    gibbon = qLowerBoundMaxValueEntropy(evaluator.model, candidate_set)(torch_X.unsqueeze(1)).detach().numpy()
    p_good = classifier.p(torch_X)

    return (gibbon * p_good).reshape(X.shape[:-1])


# these return params that maximize the objective


def MaxExpectedImprovement(evaluator, classifier, n_test=1024):
    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_improvement(evaluator, classifier, test_X).argmax())


def MaxExpectedGIBBON(evaluator, classifier, n_test=1024):
    sampler = sp.stats.qmc.Halton(d=evaluator.X.shape[-1], scramble=True)
    test_X = torch.as_tensor(sampler.random(n=n_test)).double()

    return test_X(expected_gibbon(evaluator, classifier, test_X).argmax())
