import torch
from botorch.acquisition.analytic import LogExpectedImprovement


def log_expected_sum_of_tasks_improvement(candidates, agent):
    """
    Return the expected improvement in the sum of tasks.
    """

    *input_shape, n_dim = candidates.shape
    x = torch.as_tensor(candidates.reshape(-1, 1, n_dim)).double()

    LEI = LogExpectedImprovement(
        model=agent.multimodel, best_f=agent.best_sum_of_tasks, posterior_transform=agent.scalarization
    )
    lei = LEI.forward(x).reshape(input_shape)
    log_prob = agent.dirichlet_classifier.log_prob(x).reshape(input_shape)

    return torch.clamp(lei + log_prob, min=-16)
