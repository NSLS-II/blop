import numpy as np
import scipy as sp
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy


def log_expected_sum_of_tasks_improvement(X, agent):
    """
    Return the expected improvement in the sum of tasks.
    """

    *input_shape, input_dim = np.atleast_2d(X).shape
    x = torch.tensor(X).reshape(-1, input_dim).double()

    tasks_posterior = agent.multimodel.posterior(x)

    nu = np.nanmax(agent.targets.sum(axis=-1))  # the best sum of tasks so far
    mu = agent.normalize_targets(tasks_posterior.mean.detach().numpy()).sum(axis=-1)  # the expected sum of tasks
    noise = np.array([task.regressor.likelihood.noise.item() for task in agent.tasks])
    sigma = (agent.targets.std(axis=0) * np.sqrt(tasks_posterior.variance.detach().numpy() + noise)).sum(
        axis=-1
    )  # the variance in that estimate

    log_esti = np.log(
        np.exp(-0.5 * np.square((mu - nu) / sigma)) * sigma / np.sqrt(2 * np.pi)
        + 0.5 * (mu - nu) * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))
    )

    log_prob = agent.classifier.log_prob(x).detach().numpy()

    return log_esti.reshape(input_shape) + log_prob.reshape(input_shape)


def expected_improvement(X, regressor, classifier):
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


def expected_gibbon(X, regressor, classifier, n_candidates=1024):
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
