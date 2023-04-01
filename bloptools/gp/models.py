import gpytorch
import numpy as np
import torch

from . import kernels, utils


class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, n_dof, length_scale_bounds, batch_shape=1):
        super().__init__(x, y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernels.LatentMaternKernel(
            n_dof=n_dof, length_scale_bounds=length_scale_bounds, off_diag=True
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPR:
    """
    A Gaussian process regressor, with learning methods.
    """

    def __init__(self, length_scale_bounds=(1e-3, 1e0), max_noise_fraction=1e-2):
        self.max_noise_fraction = max_noise_fraction
        self.state_dict = None
        self.length_scale_bounds = length_scale_bounds
        self.model = None

    def set_data(self, X, y):
        """
        Set the data with parameters and values.

        X: parameters
        y: function values at those parameters
        """

        if np.isnan(y).any():
            raise ValueError("One of the passed values is NaN.")

        # prepare Gaussian process ingredients for the regressor and classifier
        # use only regressable points for the regressor

        self.noise_upper_bound = 1e-1 * self.max_noise_fraction if len(y) > 1 else self.max_noise_fraction
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(0, self.noise_upper_bound)
        )

        self.model = GaussianProcessModel(
            torch.as_tensor(X).float(),
            torch.as_tensor(y).float(),
            self.likelihood,
            X.shape[-1],
            self.length_scale_bounds,
        )

        self.init_state_dict = self.model.state_dict()
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)

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
    def y(self):
        return self.torch_targets.detach().numpy().astype(float)

    @property
    def n(self):
        return self.y.size

    @property
    def n_dof(self):
        return self.X.shape[-1]

    def tell(self, X, y):
        self.model.set_train_data(
            torch.cat([self.torch_inputs, torch.as_tensor(np.atleast_2d(X))]).float(),
            torch.cat([self.torch_targets, torch.as_tensor(np.atleast_1d(y))]).float(),
            strict=False,
        )

    def train(self, training_iter=100, reuse_hypers=True, verbose=True):
        if not reuse_hypers:
            self.model.load_state_dict(self.init_state_dict)

        self.likelihood.train()
        self.model.train()

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            self.optimizer.zero_grad()
            loss = -self.mll(self.model(self.model.train_inputs[0]), self.model.train_targets)
            loss.backward()
            self.optimizer.step()

            if verbose and ((i + 1) % 100 == 0):
                print(f"{i+1}/{training_iter} inverse_length_scales: {self.model.covar_module.trans_diagonal}")

        self.state_dict = self.model.state_dict()

    def regress(self, X):
        # set to evaluation mode
        self.likelihood.eval()
        self.model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.model(torch.as_tensor(np.atleast_2d(X)).float()))

        return prediction

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPR()
        dummy.set_data(self.X, self.y)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def mean(self, X):
        *input_shape, _ = X.shape
        return self.regress(X.reshape(-1, self.n_dof)).mean.detach().numpy().reshape(input_shape)

    def sigma(self, X):
        *input_shape, _ = X.shape
        return self.regress(X.reshape(-1, self.n_dof)).stddev.detach().numpy().reshape(input_shape)

    @property
    def nu(self):
        return self.y.max()

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

    def __init__(self, length_scale_bounds=(1e-3, 1e0), **kwargs):
        self.state_dict = None
        self.length_scale_bounds = length_scale_bounds

    def set_data(self, X, c):
        """
        Set the data with parameters and values.

        X: parameters
        c: function classes at those parameters

        Passed parameters must be between [-1, 1] in every dimension. Passed values must be integer labels.
        """

        # if (x.min(axis=0) <= -1).any() or (x.max(axis=0) >= +1).any():
        #    raise ValueError('Parameters must be between -1 and +1 in each dimension.')

        self.likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(c), learn_additional_noise=True
        )

        self.model = GaussianProcessModel(
            torch.as_tensor(X).float(),
            self.likelihood.transformed_targets,
            self.likelihood,
            X.shape[-1],
            self.length_scale_bounds,
            batch_shape=2,
        )

        self.init_state_dict = self.model.state_dict()
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)

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

    def train(self, training_iter=100, reuse_hypers=True, verbose=True):
        if not reuse_hypers:
            self.model.load_state_dict(self.init_state_dict)

        self.likelihood.train()
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            self.optimizer.zero_grad()
            loss = -self.mll(self.model(self.model.train_inputs[0]), self.likelihood.transformed_targets).sum()
            loss.backward()
            self.optimizer.step()

            if verbose and ((i + 1) % 100 == 0):
                print(f"{i+1}/{training_iter} inverse_length_scales: {self.model.covar_module.trans_diagonal}")

        self.state_dict = self.model.state_dict()

    def copy(self):
        if self.model is None:
            raise RuntimeError("You cannot copy a model with no data.")

        dummy = GPC()
        dummy.set_data(self.X, self.c)
        dummy.model.load_state_dict(self.model.state_dict())

        return dummy

    def classify(self, X, return_variance=False):

        *input_shape, _ = X.shape

        x = torch.as_tensor(X.reshape(-1, self.n_dof)).float()

        # set to evaluation mode
        self.likelihood.eval()
        self.model.eval()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            dist = self.model(x)
            samples = dist.sample(torch.Size((256,))).exp()
            probabilities = (samples / samples.sum(-2, keepdim=True)).mean(0)

        return probabilities[1].detach().numpy().reshape(input_shape)
