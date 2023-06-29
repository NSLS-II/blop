import math

import gpytorch
import numpy as np
import torch


class MultiOutputLatentKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(
        self,
        num_inputs=1,
        num_outputs=1,
        off_diag=False,
        diag_prior=False,
        **kwargs,
    ):
        super(MultiOutputLatentKernel, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_off_diag = int(num_inputs * (num_inputs - 1) / 2)
        self.off_diag = off_diag

        self.nu = kwargs.get("nu", 1.5)

        # self.batch_shape = torch.Size([num_outputs])

        # output_scale_constraint = gpytorch.constraints.Positive()
        diag_params_constraint = gpytorch.constraints.Interval(1e0, 1e2)
        skew_params_constraint = gpytorch.constraints.Interval(-1e0, 1e0)

        diag_params_initial = np.sqrt(diag_params_constraint.lower_bound * diag_params_constraint.upper_bound)
        raw_diag_params_initial = diag_params_constraint.inverse_transform(diag_params_initial)

        self.register_parameter(
            name="raw_diag_params",
            parameter=torch.nn.Parameter(
                raw_diag_params_initial * torch.ones(self.num_outputs, self.num_inputs).double()
            ),
        )

        # self.register_constraint("raw_output_scale", output_scale_constraint)
        self.register_constraint("raw_diag_params", diag_params_constraint)

        if diag_prior:
            self.register_prior(
                name="diag_params_prior",
                prior=gpytorch.priors.GammaPrior(concentration=0.5, rate=0.2),
                param_or_closure=lambda m: m.diag_params,
                setting_closure=lambda m, v: m._set_diag_params(v),
            )

        if self.off_diag:
            self.register_parameter(
                name="raw_skew_params",
                parameter=torch.nn.Parameter(torch.zeros(self.num_outputs, self.n_off_diag).double()),
            )
            self.register_constraint("raw_skew_params", skew_params_constraint)

    @property
    def diag_params(self):
        return self.raw_diag_params_constraint.transform(self.raw_diag_params)

    @property
    def skew_params(self):
        return self.raw_skew_params_constraint.transform(self.raw_skew_params)

    @diag_params.setter
    def diag_params(self, value):
        self._set_diag_params(value)

    @skew_params.setter
    def skew_params(self, value):
        self._set_skew_params(value)

    def _set_skew_params(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_skew_params)
        self.initialize(raw_skew_params=self.raw_skew_params_constraint.inverse_transform(value))

    def _set_diag_params(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_diag_params)
        self.initialize(raw_diag_params=self.raw_diag_params_constraint.inverse_transform(value))

    @property
    def dimension_transform(self):
        # no rotations
        if not self.off_diag:
            T = torch.eye(self.num_inputs, dtype=torch.float64)

        # construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
        else:
            A = torch.zeros((self.num_outputs, self.num_inputs, self.num_inputs), dtype=torch.float64)
            upper_indices = np.triu_indices(self.num_inputs, k=1)
            for output_index in range(self.num_outputs):
                A[(output_index, *upper_indices)] = self.skew_params[output_index]
            A += -A.transpose(-1, -2)
            T = torch.linalg.matrix_exp(A)

        diagonal_transform = torch.cat([torch.diag(_values).unsqueeze(0) for _values in self.diag_params], dim=0)
        T = torch.matmul(diagonal_transform, T)

        return T

    def forward(self, x1, x2, diag=False, **params):
        # adapted from the Matern kernel
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

        trans_x1 = torch.matmul(self.dimension_transform.unsqueeze(1), (x1 - mean).unsqueeze(-1)).squeeze(-1)
        trans_x2 = torch.matmul(self.dimension_transform.unsqueeze(1), (x2 - mean).unsqueeze(-1)).squeeze(-1)

        distance = self.covar_dist(trans_x1, trans_x2, diag=diag, **params)

        # if distance.shape[0] == 1:
        #     distance = distance.squeeze(0)  # this is extremely necessary

        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)

        return constant_component * exp_component


class LatentMaternKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        n_dim,
        off_diag=False,
        diagonal_prior=False,
        **kwargs,
    ):
        super(LatentMaternKernel, self).__init__()

        self.n_dim = n_dim
        self.n_off_diag = int(n_dim * (n_dim - 1) / 2)
        self.off_diag = off_diag

        # output_scale_constraint = gpytorch.constraints.Positive()
        diag_params_constraint = gpytorch.constraints.Interval(1e-1, 1e2)
        skew_params_constraint = gpytorch.constraints.Interval(-1e0, 1e0)

        diag_params_initial = np.sqrt(diag_params_constraint.lower_bound * diag_params_constraint.upper_bound)
        raw_diag_params_initial = diag_params_constraint.inverse_transform(diag_params_initial)

        # self.register_parameter(
        #    name="raw_output_scale", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1).double())
        # )
        self.register_parameter(
            name="raw_diag_params",
            parameter=torch.nn.Parameter(
                raw_diag_params_initial * torch.ones(*self.batch_shape, self.n_dim).double()
            ),
        )

        # self.register_constraint("raw_output_scale", output_scale_constraint)
        self.register_constraint("raw_diag_params", diag_params_constraint)

        if diagonal_prior:
            self.register_prior(
                name="diag_params_prior",
                prior=gpytorch.priors.GammaPrior(concentration=0.5, rate=0.2),
                param_or_closure=lambda m: m.diag_params,
                setting_closure=lambda m, v: m._set_diag_params(v),
            )

        if self.off_diag:
            self.register_parameter(
                name="raw_skew_params",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.n_off_diag).double()),
            )
            self.register_constraint("raw_skew_params", skew_params_constraint)

    # @property
    # def output_scale(self):
    #     return self.raw_output_scale_constraint.transform(self.raw_output_scale)

    @property
    def diag_params(self):
        return self.raw_diag_params_constraint.transform(self.raw_diag_params)

    @property
    def skew_params(self):
        return self.raw_skew_params_constraint.transform(self.raw_skew_params)

    # @output_scale.setter
    # def output_scale(self, value):
    #     self._set_output_scale(value)

    @diag_params.setter
    def diag_params(self, value):
        self._set_diag_params(value)

    @skew_params.setter
    def skew_params(self, value):
        self._set_skew_params(value)

    def _set_skew_params(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_skew_params)
        self.initialize(raw_skew_params=self.raw_skew_params_constraint.inverse_transform(value))

    # def _set_output_scale(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.raw_output_scale)
    #     self.initialize(raw_output_scale=self.raw_output_scale_constraint.inverse_transform(value))

    def _set_diag_params(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_diag_params)
        self.initialize(raw_diag_params=self.raw_diag_params_constraint.inverse_transform(value))

    @property
    def trans_matrix(self):
        # no rotations
        if not self.off_diag:
            T = torch.eye(self.n_dim).double()

        # construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
        else:
            A = torch.zeros((self.n_dim, self.n_dim)).double()
            A[np.triu_indices(self.n_dim, k=1)] = self.skew_params
            A += -A.T
            T = torch.linalg.matrix_exp(A)

        T = torch.matmul(torch.diag(self.diag_params), T)

        return T

    def forward(self, x1, x2=None, diag=False, auto=False, last_dim_is_batch=False, **params):
        # returns the homoskedastic diagonal
        if diag:
            # return torch.square(self.output_scale[0]) * torch.ones((*self.batch_shape, *x1.shape[:-1]))
            return torch.ones((*self.batch_shape, *x1.shape[:-1]))

        # computes the autocovariance of the process at the parameters
        if auto:
            x2 = x1

        # print(x1, x2)

        # x1 and x2 are arrays of shape (..., n_1, n_dim) and (..., n_2, n_dim)
        _x1, _x2 = torch.as_tensor(x1).double(), torch.as_tensor(x2).double()

        # dx has shape (..., n_1, n_2, n_dim)
        dx = _x1.unsqueeze(-2) - _x2.unsqueeze(-3)

        # transform coordinates with hyperparameters (this applies lengthscale and rotations)
        trans_dx = torch.matmul(self.trans_matrix, dx.unsqueeze(-1))

        # total transformed distance. D has shape (..., n_1, n_2)
        d_eff = torch.sqrt(torch.matmul(trans_dx.transpose(-1, -2), trans_dx).sum((-1, -2)) + 1e-12)

        # Matern covariance of effective order nu=3/2.
        # nu=3/2 is a special case and has a concise closed-form expression
        # In general, this is something between an exponential (n=1/2) and a Gaussian (n=infinity)
        # https://en.wikipedia.org/wiki/Matern_covariance_function

        # C = torch.exp(-d_eff) # Matern_0.5 (exponential)
        C = (1 + d_eff) * torch.exp(-d_eff)  # Matern_1.5
        # C = (1 + d_eff + 1 / 3 * torch.square(d_eff)) * torch.exp(-d_eff)  # Matern_2.5
        # C = torch.exp(-0.5 * np.square(d_eff)) # Matern_infinity (RBF)

        # C = torch.square(self.output_scale[0]) * torch.exp(-torch.square(d_eff))

        # print(f'{diag = } {C.shape = }')

        return C
