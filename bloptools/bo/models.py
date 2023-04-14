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
            kernels.LatentMaternKernel(n_dof=train_X.shape[-1], off_diag=True), num_tasks=self._num_outputs, rank=1
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
        self.covar_module = kernels.LatentMaternKernel(n_dof=train_X.shape[-1], off_diag=True)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR:

    """
    A Gaussian process regressor, with learning methods.
    """

    def __init__(self, MIN_SNR=1e-2):
        self.model = None
        self.MIN_SNR = MIN_SNR

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

        self.num_tasks = Y.shape[-1]

        self.noise_upper_bound = np.square((1 if not Y.shape[0] > 1 else Y.std(axis=0)) / self.MIN_SNR)

        likelihood_noise_constraint = gpytorch.constraints.Interval(
            0, torch.as_tensor(self.noise_upper_bound).double()
        )

        likelihood = MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks,
            noise_constraint=likelihood_noise_constraint,
        ).double()

        self.model = BoTorchMultiTaskGP(
            train_X=torch.as_tensor(X).double(),
            train_Y=torch.as_tensor(Y).double(),
            likelihood=likelihood,
        ).double()

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    @property
    def torch_inputs(self):
        return self.model.train_inputs[0]

    @property
    def torch_targets(self):
        return self.model.train_targets

    @property
    def X(self):
        return self.torch_inputs.detach().numpy().astype(float)

    @property
    def Y(self):
        return self.torch_targets.detach().numpy().astype(float)

    @property
    def n(self):
        return self.Y.size

    @property
    def n_dof(self):
        return self.X.shape[-1]

    def tell(self, X, Y):
        self.model.set_train_data(
            torch.cat([self.torch_inputs, torch.as_tensor(np.atleast_2d(X))]).double(),
            torch.cat([self.torch_targets, torch.as_tensor(np.atleast_2d(Y))]).double(),
            strict=False,
        )

    def train(self, maxiter=100, lr=1e-2):
        botorch.optim.fit.fit_gpytorch_mll_torch(self.mll, optimizer=torch.optim.Adam, step_limit=256)

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPR()
        dummy.set_data(self.X, self.Y)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def mean(self, X):
        *input_shape, _ = X.shape
        x = torch.as_tensor(X.reshape(-1, self.n_dof)).double()
        return self.model.posterior(x).mean.detach().numpy().reshape(input_shape)

    def sigma(self, X):
        *input_shape, _ = X.shape
        x = torch.as_tensor(X.reshape(-1, self.n_dof)).double()
        return self.model.posterior(x).stddev.detach().numpy().reshape(input_shape)

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


class GPC:
    """
    A Gaussian process classifier, with learning methods.
    """

    def __init__(self):
        self.model = None

    def set_data(self, X, c):
        """
        Set the data with parameters and values.

        X: parameters
        c: function classes at those parameters

        Passed parameters must be between [-1, 1] in every dimension. Passed values must be integer labels.
        """

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(c).int(), learn_additional_noise=True
        ).double()

        self.model = BoTorchClassifier(
            torch.as_tensor(X).double(),
            dirichlet_likelihood.transformed_targets,
            dirichlet_likelihood,
        ).double()

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    @property
    def torch_inputs(self):
        return self.model.train_inputs[0]

    @property
    def torch_targets(self):
        return self.model.train_targets

    @property
    def X(self):
        return self.torch_inputs.detach().numpy().astype(float)

    @property
    def c(self):
        return self.torch_targets.detach().numpy().argmax(axis=0)

    @property
    def n(self):
        return self.c.size

    @property
    def n_dof(self):
        return self.X.shape[-1]

    def tell(self, X, c):
        self.set_data(np.r_[self.X, np.atleast_2d(X)], np.r_[self.c, np.atleast_1d(c)])

    def train(self, maxiter=100, lr=1e-2):
        botorch.optim.fit.fit_gpytorch_mll_torch(self.mll, optimizer=torch.optim.Adam, step_limit=256)

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPC()
        dummy.set_data(self.X, self.c)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def p(self, X, n_samples=256):
        *input_shape, _ = X.shape

        x = torch.as_tensor(X.reshape(-1, self.n_dof)).double()

        samples = self.model.posterior(x).sample(torch.Size((n_samples,))).exp()

        return (samples / samples.sum(-3, keepdim=True)).mean(0)[1].reshape(input_shape).detach().numpy()

    def entropy(self, X, n_samples=256):
        p = self.p(X, n_samples)
        q = 1 - p

        return -p * np.log(p) - q * np.log(q)
