import torch
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement


class Acquisition:
    def __init__(self, *args, **kwargs):
        ...

    @staticmethod
    def log_expected_improvement(candidates, agent):
        *input_shape, n_dim = candidates.shape

        x = torch.as_tensor(candidates.reshape(-1, 1, n_dim)).double()

        LEI = LogExpectedImprovement(
            model=agent.task_model, best_f=agent.best_sum_of_tasks, posterior_transform=agent.scalarization
        )

        lei = LEI.forward(x)
        feas_log_prob = agent.feas_model(x)

        return (lei.reshape(input_shape) + feas_log_prob.reshape(input_shape)).detach().numpy()

    @staticmethod
    def log_probability_of_improvement(candidates, agent):
        *input_shape, n_dim = candidates.shape

        x = torch.as_tensor(candidates.reshape(-1, 1, n_dim)).double()

        LPI = LogProbabilityOfImprovement(
            model=agent.task_model, best_f=agent.best_sum_of_tasks, posterior_transform=agent.scalarization
        )

        lpi = LPI.forward(x)
        feas_log_prob = agent.feas_model(x)

        return (lpi.reshape(input_shape) + feas_log_prob.reshape(input_shape)).detach().numpy()
