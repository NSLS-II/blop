import gpytorch
import numpy as np
import torch


class LatentMaternKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        n_dof,
        off_diag=True,
        **kwargs,
    ):
        super(LatentMaternKernel, self).__init__()

        self.n_dof = n_dof
        self.n_off_diag = int(n_dof * (n_dof - 1) / 2)
        self.off_diag = off_diag

        # output_scale_constraint = gpytorch.constraints.Positive()
        trans_diagonal_constraint = gpytorch.constraints.Interval(2e0, 2e1)
        trans_off_diag_constraint = gpytorch.constraints.Interval(-1e0, 1e0)

        trans_diagonal_initial = np.sqrt(
            trans_diagonal_constraint.lower_bound * trans_diagonal_constraint.upper_bound
        )
        raw_trans_diagonal_initial = trans_diagonal_constraint.inverse_transform(trans_diagonal_initial)

        # self.register_parameter(
        #    name="raw_output_scale", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1).double())
        # )
        self.register_parameter(
            name="raw_trans_diagonal",
            parameter=torch.nn.Parameter(
                raw_trans_diagonal_initial * torch.ones(*self.batch_shape, self.n_dof).double()
            ),
        )

        # self.register_constraint("raw_output_scale", output_scale_constraint)
        self.register_constraint("raw_trans_diagonal", trans_diagonal_constraint)

        if self.off_diag:
            self.register_parameter(
                name="raw_trans_off_diag",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.n_off_diag).double()),
            )
            self.register_constraint("raw_trans_off_diag", trans_off_diag_constraint)

    # @property
    # def output_scale(self):
    #     return self.raw_output_scale_constraint.transform(self.raw_output_scale)

    @property
    def trans_diagonal(self):
        return self.raw_trans_diagonal_constraint.transform(self.raw_trans_diagonal)

    @property
    def trans_off_diag(self):
        return self.raw_trans_off_diag_constraint.transform(self.raw_trans_off_diag)

    # @output_scale.setter
    # def output_scale(self, value):
    #     self._set_output_scale(value)

    @trans_diagonal.setter
    def trans_diagonal(self, value):
        self._set_trans_diagonal(value)

    @trans_off_diag.setter
    def trans_off_diag(self, value):
        self._set_trans_off_diag(value)

    def _set_trans_off_diag(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_trans_off_diag)
        self.initialize(raw_trans_off_diag=self.raw_trans_off_diag_constraint.inverse_transform(value))

    # def _set_output_scale(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.raw_output_scale)
    #     self.initialize(raw_output_scale=self.raw_output_scale_constraint.inverse_transform(value))

    def _set_trans_diagonal(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_trans_diagonal)
        self.initialize(raw_trans_diagonal=self.raw_trans_diagonal_constraint.inverse_transform(value))

    @property
    def trans_matrix(self):
        # no rotations
        if not self.off_diag:
            T = torch.eye(self.n_dof).double()

        # construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
        else:
            A = torch.zeros((self.n_dof, self.n_dof)).double()
            A[np.triu_indices(self.n_dof, k=1)] = self.trans_off_diag
            A += -A.T
            T = torch.linalg.matrix_exp(A)

        T = torch.matmul(torch.diag(self.trans_diagonal), T)

        return T

    def forward(self, x1, x2=None, diag=False, auto=False, last_dim_is_batch=False, **params):
        # returns the homoskedastic diagonal
        if diag:
            # return torch.square(self.output_scale[0]) * torch.ones((*self.batch_shape, *x1.shape[:-1]))
            return torch.ones((*self.batch_shape, *x1.shape[:-1]))

        # computes the autocovariance of the process at the parameters
        if auto:
            x2 = x1

        # x1 and x2 are arrays of shape (..., n_1, n_dof) and (..., n_2, n_dof)
        _x1, _x2 = torch.as_tensor(x1).double(), torch.as_tensor(x2).double()

        # dx has shape (..., n_1, n_2, n_dof)
        dx = _x1.unsqueeze(-2) - _x2.unsqueeze(-3)

        # transform coordinates with hyperparameters (this applies lengthscale and rotations)
        trans_dx = torch.matmul(self.trans_matrix, dx.unsqueeze(-1))

        # total transformed distance. D has shape (..., n_1, n_2)
        d_eff = torch.sqrt(torch.matmul(trans_dx.transpose(-1, -2), trans_dx).sum((-1, -2)) + 1e-12)

        # Matern covariance of effective order nu=3/2.
        # nu=3/2 is a special case and has a concise closed-form expression
        # In general, this is something between an exponential (n=1/2) and a Gaussian (n=infinity)
        # https://en.wikipedia.org/wiki/Matern_covariance_function
        C = (1 + d_eff) * torch.exp(-d_eff)

        # C = torch.square(self.output_scale[0]) * torch.exp(-torch.square(d_eff))

        # print(f'{diag = } {C.shape = }')

        return C
