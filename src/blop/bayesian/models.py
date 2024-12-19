import botorch
import gpytorch
import torch
from botorch.models.gp_regression import SingleTaskGP

from . import kernels


def train_model(model, hypers=None, max_fails=4, **kwargs):
    """Fit all of the agent's models. All kwargs are passed to `botorch.fit.fit_gpytorch_mll`."""
    fails = 0
    while True:
        try:
            if hypers is not None:
                model.load_state_dict(hypers)
            else:
                botorch.fit.fit_gpytorch_mll(gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model), **kwargs)
            model.trained = True
            return
        except Exception as e:
            if fails < max_fails:
                fails += 1
            else:
                raise e


def construct_single_task_model(X, y, skew_dims=None, min_noise=1e-6, max_noise=1e0):
    """
    Construct an untrained model for an objective.
    """

    skew_dims = skew_dims if skew_dims is not None else [(i,) for i in range(X.shape[-1])]

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(
            torch.tensor(min_noise),
            torch.tensor(max_noise),
        ),
    )

    input_transform = botorch.models.transforms.input.Normalize(d=X.shape[-1])
    outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)  # , batch_shape=torch.Size((1,)))

    if not X.isfinite().all():
        raise ValueError("'X' must not contain points that are inf or NaN.")
    if not y.isfinite().all():
        raise ValueError("'y' must not contain points that are inf or NaN.")

    return LatentGP(
        train_inputs=X,
        train_targets=y,
        likelihood=likelihood,
        skew_dims=skew_dims,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )


class LatentGP(SingleTaskGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, *args, **kwargs):
        super().__init__(train_inputs, train_targets, *args, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

        self.covar_module = kernels.LatentKernel(
            num_inputs=train_inputs.shape[-1],
            num_outputs=train_targets.shape[-1],
            skew_dims=skew_dims,
            priors=True,
            scale=True,
            **kwargs,
        )

        self.trained = False


class LatentConstraintModel(LatentGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, *args, **kwargs):
        super().__init__(train_inputs, train_targets, skew_dims, *args, **kwargs)

        self.trained = False

    def fitness(self, x, n_samples=1024):
        """
        Takes in a (..., m) dimension tensor and returns a (..., n_classes) tensor
        """
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)


class LatentDirichletClassifier(LatentGP):
    def __init__(self, train_inputs, train_targets, skew_dims=True, *args, **kwargs):
        super().__init__(train_inputs, train_targets, skew_dims, *args, **kwargs)

        self.trained = False

    def probabilities(self, x, n_samples=1024):
        """
        Takes in a (..., m) dimension tensor and returns a (..., n_classes) tensor
        """
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)
