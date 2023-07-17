import botorch
import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP

from . import kernels


class LatentDirichletClassifier(botorch.models.gp_regression.SingleTaskGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, batch_dimension=None, *args, **kwargs):
        super().__init__(train_inputs, train_targets, *args, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernels.LatentKernel(
            num_inputs=train_inputs.shape[-1],
            num_outputs=train_targets.shape[-1],
            skew_dims=skew_dims,
            diag_prior=True,
            scale=True,
            batch_dimension=batch_dimension,
            **kwargs
        )

    def log_prob(self, x, n_samples=256):
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return torch.log((samples / samples.sum(-1, keepdim=True)).mean(0)[:, 1]).reshape(*input_shape, 1)


class LatentGP(botorch.models.gp_regression.SingleTaskGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, batch_dimension=None, *args, **kwargs):
        super().__init__(train_inputs, train_targets, *args, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

        self.covar_module = kernels.LatentKernel(
            num_inputs=train_inputs.shape[-1],
            num_outputs=train_targets.shape[-1],
            skew_dims=skew_dims,
            diag_prior=True,
            scale=True,
            batch_dimension=batch_dimension,
            **kwargs
        )


class OldBoTorchSingleTaskGP(ExactGP, GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(OldBoTorchSingleTaskGP, self).__init__(train_inputs, train_targets, likelihood)
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


class OldBoTorchDirichletClassifier(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_inputs, train_targets, likelihood):
        super(OldBoTorchDirichletClassifier, self).__init__(train_inputs, train_targets, likelihood)
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
