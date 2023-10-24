import botorch
import gpytorch
import torch

from . import kernels


class LatentGP(botorch.models.gp_regression.SingleTaskGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, *args, **kwargs):
        super().__init__(train_inputs, train_targets, *args, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

        self.covar_module = kernels.LatentKernel(
            num_inputs=train_inputs.shape[-1],
            num_outputs=train_targets.shape[-1],
            skew_dims=skew_dims,
            diag_prior=True,
            scale=True,
            **kwargs
        )


class LatentDirichletClassifier(LatentGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, *args, **kwargs):
        super().__init__(train_inputs, train_targets, skew_dims, *args, **kwargs)

    def probabilities(self, x, n_samples=1024):
        """
        Takes in a (..., m) dimension tensor and returns a (..., n_classes) tensor
        """
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)
