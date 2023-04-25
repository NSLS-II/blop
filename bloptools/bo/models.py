import botorch
import gpytorch
import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models import ExactGP

from .. import utils
from . import kernels


class BoTorchMultiTaskGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        self._num_outputs = train_Y.shape[-1]

        super(BoTorchMultiTaskGP, self).__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=self._num_outputs)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernels.LatentMaternKernel(n_dof=train_X.shape[-1], off_diag=True),
            num_tasks=self._num_outputs,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BoTorchClassifier(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        super(BoTorchClassifier, self).__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=len(train_Y.unique()))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernels.LatentMaternKernel(n_dof=train_X.shape[-1], off_diag=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BoTorchModelWrapper:
    def __init__(self, bounds, MIN_SNR=1e-2):
        self.model = None
        self.bounds = bounds
        self.MIN_SNR = MIN_SNR

        self.prepare_inputs = lambda inputs: torch.tensor(
            (inputs - self.bounds.min(axis=1)) / self.bounds.ptp(axis=1)
        ).double()
        self.unprepare_inputs = lambda inputs: inputs.detach().numpy() * self.bounds.ptp(axis=1) + self.bounds.min(
            axis=1
        )

    def tell(self, X, Y):
        self.model.set_train_data(
            torch.cat([self.train_inputs, self.prepare_inputs(X)]),
            torch.cat([self.train_targets, self.prepare_targets(Y)]),
            strict=False,
        )

    def train(self, **kwargs):
        botorch.optim.fit.fit_gpytorch_mll_torch(self.mll, optimizer=torch.optim.Adam, **kwargs)

    @property
    def train_inputs(self):
        return self.prepare_inputs(self.X)

    @property
    def train_targets(self):
        return self.prepare_targets(self.Y)

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def n_dof(self):
        return self.X.shape[-1]

    @property
    def n_tasks(self):
        return self.Y.shape[-1]


class GPR(BoTorchModelWrapper):

    """
    A Gaussian process regressor, with learning methods.
    """

    def set_data(self, X, Y):
        """
        Instantiate the GP with parameters and values.

        X: parameters
        Y: values of functions at those parameters
        """

        if np.isnan(Y).any():
            raise ValueError("One of the passed values is NaN.")

        # prepare Gaussian process ingredients for the regressor and classifier
        # use only regressable points for the regressor

        self.X, self.Y = X, Y

        self.target_means = Y.mean(axis=0)
        self.target_scales = Y.std(axis=0)

        self.prepare_targets = lambda targets: torch.tensor(
            (targets - self.target_means[None]) / self.target_scales[None]
        ).double()
        self.unprepare_targets = (
            lambda targets: targets.detach().numpy() * self.target_scales[None] + self.target_means[None]
        )

        self.num_tasks = Y.shape[-1]

        self.noise_upper_bound = np.square(1 / self.MIN_SNR)

        likelihood_noise_constraint = gpytorch.constraints.Interval(
            0, torch.tensor(self.noise_upper_bound).double()
        )

        likelihood = MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks,
            noise_constraint=likelihood_noise_constraint,
        ).double()

        self.model = BoTorchMultiTaskGP(
            train_X=self.train_inputs,
            train_Y=self.train_targets,
            likelihood=likelihood,
        ).double()

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def tell(self, X, Y):
        self.set_data(np.r_[self.X, np.atleast_2d(X)], np.r_[self.Y, np.atleast_2d(Y)])

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPR(bounds=self.bounds, MIN_SNR=self.MIN_SNR)
        dummy.set_data(self.X, self.Y)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def mean(self, X):
        *input_shape, _ = X.shape
        x = self.prepare_inputs(X).reshape(-1, self.n_dof)
        return self.unprepare_targets(self.model.posterior(x).mean).reshape(input_shape)

    def sigma(self, X):
        *input_shape, _ = X.shape
        x = self.prepare_inputs(X).reshape(-1, self.n_dof)
        return self.target_scales * self.model.posterior(x).stddev.detach().numpy().reshape(input_shape)

    @property
    def scale(self):
        return self.model.covar_module.task_covar_module.var.sqrt().item()

    @property
    def nu(self):
        return self.Y.max()

    def entropy(self, X):
        return 0.5 + np.log(np.sqrt(2 * np.pi) * self.sigma(X))

    def normalized_entropy(self, X):
        return 0.5 + np.log(np.sqrt(2 * np.pi) * self.sigma(X) / self.scale)

    def _contingent_fisher_information_matrix(self, test_X, delta=1e-3):
        x = test_X.reshape(-1, self.n_dof)

        total_X = np.r_[[np.r_[self.X, _x[None]] for _x in x]]
        (n_sets, n_per_set, n_dof) = total_X.shape

        # both of these have shape (n_hypers, n_sets, n_per_set, n_per_set)

        dummy_evaluator = self.copy()
        C = dummy_evaluator.model.covar_module.forward(total_X, total_X).detach().numpy().astype(float)
        C += dummy_evaluator.likelihood.noise.item() * np.eye(n_per_set)[None]

        M = dummy_evaluator.model.mean_module.forward(total_X).detach().numpy().astype(float)[:, :, None]

        delta = 1e-3

        dC_dtheta = np.zeros((0, n_sets, n_per_set, n_per_set))
        dM_dtheta = np.zeros((0, n_sets, n_per_set, 1))

        for hyper_name, hyper in self.model.named_hyperparameters():
            for hyper_dim, hyper_val in enumerate(hyper.detach().numpy()):
                dummy_state_dict = dummy_evaluator.model.state_dict().copy()
                # print(dummy_evaluator.likelihood.noise.item())
                dummy_state_dict[hyper_name][hyper_dim] += delta
                # print(dummy_evaluator.likelihood.noise.item())
                dummy_evaluator.model.load_state_dict(dummy_state_dict)
                C_ = dummy_evaluator.model.covar_module.forward(total_X, total_X).detach().numpy().astype(float)
                M_ = dummy_evaluator.model.mean_module.forward(total_X).detach().numpy().astype(float)[:, :, None]
                C_ += dummy_evaluator.likelihood.noise.item() * np.eye(n_per_set)[None]
                dummy_evaluator.model.load_state_dict(self.model.state_dict())
                dC_dtheta = np.r_[dC_dtheta, (C_ - C)[None] / delta]
                dM_dtheta = np.r_[dM_dtheta, (M_ - M)[None] / delta]

        n_hypers = len(dC_dtheta)
        invC = np.linalg.inv(C)

        FIM_stack = np.zeros((n_sets, n_hypers, n_hypers))

        for i in range(n_hypers):
            for j in range(n_hypers):
                FIM_stack[:, i, j] = utils.mprod(
                    np.swapaxes(dM_dtheta[i], -1, -2), invC, dM_dtheta[j]
                ).ravel() + 0.5 * np.trace(utils.mprod(invC, dC_dtheta[i], invC, dC_dtheta[j]), axis1=-1, axis2=-2)

        return FIM_stack


class GPC(BoTorchModelWrapper):
    """
    A Gaussian process classifier, with learning methods.
    """

    def set_data(self, X, c):
        """
        Set the data with parameters and values.

        X: parameters
        c: function classes at those parameters

        Passed parameters must be between [-1, 1] in every dimension. Passed values must be integer labels.
        """

        self.X, self.c = X, c

        self.prepare_targets = lambda targets: torch.tensor(targets).int()
        self.unprepare_targets = lambda targets: targets.detach().numpy().astype(int)

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(c).int(), learn_additional_noise=True
        ).double()

        self.model = BoTorchClassifier(
            self.train_inputs,
            dirichlet_likelihood.transformed_targets,
            dirichlet_likelihood,
        ).double()

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def tell(self, X, c):
        self.set_data(np.r_[self.X, np.atleast_2d(X)], np.r_[self.c, np.atleast_1d(c)])

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPC(bounds=self.bounds, MIN_SNR=self.MIN_SNR)
        dummy.set_data(self.X, self.c)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def p(self, X, n_samples=256):
        *input_shape, _ = X.shape
        x = self.prepare_inputs(X).reshape(-1, self.n_dof)
        samples = self.model.posterior(x).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-3, keepdim=True)).mean(0)[1].reshape(input_shape).detach().numpy()

    def entropy(self, X, n_samples=256):
        p = self.p(X, n_samples)
        q = 1 - p
        return -p * np.log(p) - q * np.log(q)
