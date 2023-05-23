import botorch
import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP

from . import kernels


class BoTorchSingleTaskGP(ExactGP, GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(BoTorchSingleTaskGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernels.LatentMaternKernel(n_dim=train_inputs.shape[-1], off_diag=True, diagonal_prior=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BoTorchMultiTaskGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_inputs, train_targets, likelihood):
        self._num_outputs = train_targets.shape[-1]

        super(BoTorchMultiTaskGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=self._num_outputs)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernels.LatentMaternKernel(n_dim=train_inputs.shape[-1], off_diag=True, diagonal_prior=True),
            num_tasks=self._num_outputs,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BoTorchDirichletClassifier(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_inputs, train_targets, likelihood):
        super(BoTorchDirichletClassifier, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=len(train_targets.unique()))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernels.LatentMaternKernel(n_dim=train_inputs.shape[-1], off_diag=False, diagonal_prior=False)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def log_prob(self, x, n_samples=256):
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return torch.log((samples / samples.sum(-3, keepdim=True)).mean(0)[1]).reshape(*input_shape)
