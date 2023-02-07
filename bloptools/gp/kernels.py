import torch
import gpytorch
import numpy as np
import scipy as sp

#gpytorch.linear_operator.operators import LinearOperator, MatmulLinearOperator, RootLinearOperator

class LatentMaternKernel(gpytorch.kernels.Kernel):

    def __init__(
        self,
        n_dof,
        off_diag=True,
        **kwargs,
    ):
        super(LatentMaternKernel, self).__init__(**kwargs)
        
        self.n_dof      = n_dof
        self.n_off_diag = int(n_dof * (n_dof - 1) / 2)
        self.off_diag   = off_diag
        
        outputscale_constraint    = gpytorch.constraints.Positive()
        trans_diagonal_constraint = gpytorch.constraints.Interval(1e0, 1e2)
        trans_off_diag_constraint = gpytorch.constraints.Interval(-1e1, 1e1)
        
        trans_diagonal_initial = np.mean([trans_diagonal_constraint.lower_bound, trans_diagonal_constraint.upper_bound])
        raw_trans_diagonal_initial = trans_diagonal_constraint.inverse_transform(trans_diagonal_initial)

        self.register_parameter(name="raw_outputscale",    parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1)))
        self.register_parameter(name="raw_trans_diagonal", parameter=torch.nn.Parameter(raw_trans_diagonal_initial * torch.ones(*self.batch_shape, self.n_dof)))
        
        self.register_constraint("raw_outputscale",    outputscale_constraint)
        self.register_constraint("raw_trans_diagonal", trans_diagonal_constraint)
        
        if self.off_diag:
            self.register_parameter(name="raw_trans_off_diag", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.n_off_diag)))
            self.register_constraint("raw_trans_off_diag", trans_off_diag_constraint)        
        
    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @property
    def trans_diagonal(self):
        return self.raw_trans_diagonal_constraint.transform(self.raw_trans_diagonal)
    
    @property
    def trans_off_diag(self):
        return self.raw_trans_off_diag_constraint.transform(self.raw_trans_off_diag)
    
    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

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
        
    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def _set_trans_diagonal(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_trans_diagonal)
        self.initialize(raw_trans_diagonal=self.raw_trans_diagonal_constraint.inverse_transform(value))
        
    @property
    def trans_matrix(self):
        T = torch.diag(self.trans_diagonal)
        if self.off_diag: T[np.triu_indices(self.n_dof, k=1)] = self.trans_off_diag
        return T
    
    def forward(
        self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        
        # x1 and x2 are arrays of shape (..., n_1, n_dof) and (..., n_2, n_dof)
        _x1, _x2 = torch.as_tensor(x1).float(), torch.as_tensor(x2).float()
        
        # dx has shape (..., n_1, n_2, n_dof)
        dx = _x1.unsqueeze(-2) - _x2.unsqueeze(-3)

        # transform coordinate frame with hyperparameters (this applies lengthscale and rotations)
        trans_dx = torch.matmul(self.trans_matrix, dx.unsqueeze(-1))
        
        # total transformed distance. D has shape (..., n_1, n_2)
        D = torch.sqrt(torch.matmul(trans_dx.transpose(-1,-2), trans_dx).sum((-1,-2)) + 1e-12)

        # Matern covariance of effective order nu=3/2. 
        # nu=3/2 is a special case and has a concise closed-form expression
        # In general, this is something between an exponential (n=1/2) and a Gaussian (n=infinity) 
        # https://en.wikipedia.org/wiki/Matern_covariance_function
        return self.outputscale[0] * (1 + D) * torch.exp(-D) 
    