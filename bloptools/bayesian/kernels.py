import gpytorch
import numpy as np
import torch


class LatentKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    num_outputs = 1
    batch_inverse_lengthscale = 1e6

    def __init__(
        self,
        num_inputs=1,
        skew_dims=True,
        diag_prior=True,
        scale_output=True,
        **kwargs,
    ):
        super(LatentKernel, self).__init__()

        self.num_inputs = num_inputs
        self.scale_output = scale_output

        self.nu = kwargs.get("nu", 2.5)

        if type(skew_dims) is bool:
            if skew_dims:
                self.skew_dims = [torch.arange(self.num_inputs)]
            else:
                self.skew_dims = [torch.arange(0)]
        elif hasattr(skew_dims, "__iter__"):
            self.skew_dims = [torch.tensor(np.atleast_1d(skew_group)) for skew_group in skew_dims]
        else:
            raise ValueError('arg "skew_dims" must be True, False, or an iterable of tuples of ints.')

        # if not all([len(skew_group) >= 2 for skew_group in self.skew_dims]):
        #     raise ValueError("must have at least two dims per skew group")
        skewed_dims = [dim for skew_group in self.skew_dims for dim in skew_group]
        if not len(set(skewed_dims)) == len(skewed_dims):
            raise ValueError("values in skew_dims must be unique")
        if not max(skewed_dims) < self.num_inputs:
            raise ValueError("invalud dimension index in skew_dims")

        skew_group_submatrix_indices = []
        for dim in range(self.num_outputs):
            for skew_group in self.skew_dims:
                j, k = skew_group[torch.triu_indices(len(skew_group), len(skew_group), 1)].unsqueeze(1)
                i = dim * torch.ones(j.shape).long()
                skew_group_submatrix_indices.append(torch.cat((i, j, k), dim=0))

        self.diag_matrix_indices = tuple(
            [
                torch.kron(torch.arange(self.num_outputs), torch.ones(self.num_inputs)).long(),
                *2 * [torch.arange(self.num_inputs).repeat(self.num_outputs)],
            ]
        )

        self.skew_matrix_indices = (
            tuple(torch.cat(skew_group_submatrix_indices, dim=1))
            if len(skew_group_submatrix_indices) > 0
            else tuple([[], []])
        )

        self.n_skew_entries = len(self.skew_matrix_indices[0])

        diag_entries_constraint = gpytorch.constraints.Positive()
        raw_diag_entries_initial = (
            diag_entries_constraint.inverse_transform(torch.tensor(1e-1))
            * torch.ones(self.num_outputs, self.num_inputs).double()
        )

        self.register_parameter(name="raw_diag_entries", parameter=torch.nn.Parameter(raw_diag_entries_initial))
        self.register_constraint(param_name="raw_diag_entries", constraint=diag_entries_constraint)

        if diag_prior:
            self.register_prior(
                name="diag_entries_prior",
                prior=gpytorch.priors.GammaPrior(concentration=2, rate=1),
                param_or_closure=lambda m: m.diag_entries,
                setting_closure=lambda m, v: m._set_diag_entries(v),
            )

        if self.n_skew_entries > 0:
            skew_entries_constraint = gpytorch.constraints.Interval(-1e0, 1e0)
            skew_entries_initial = torch.zeros((self.num_outputs, self.n_skew_entries), dtype=torch.float64)
            self.register_parameter(name="raw_skew_entries", parameter=torch.nn.Parameter(skew_entries_initial))
            self.register_constraint(param_name="raw_skew_entries", constraint=skew_entries_constraint)

        if self.scale_output:
            outputscale_constraint = gpytorch.constraints.Positive()
            outputscale_prior = gpytorch.priors.GammaPrior(concentration=2, rate=0.15)

            self.register_parameter(
                name="raw_outputscale",
                parameter=torch.nn.Parameter(torch.ones(1)),
            )

            self.register_constraint("raw_outputscale", constraint=outputscale_constraint)

            self.register_prior(
                name="outputscale_prior",
                prior=outputscale_prior,
                param_or_closure=lambda m: m.outputscale,
                setting_closure=lambda m, v: m._set_outputscale(v),
            )

    @property
    def diag_entries(self):
        return self.raw_diag_entries_constraint.transform(self.raw_diag_entries)

    @property
    def skew_entries(self):
        return self.raw_skew_entries_constraint.transform(self.raw_skew_entries)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @diag_entries.setter
    def diag_entries(self, value):
        self._set_diag_entries(value)

    @skew_entries.setter
    def skew_entries(self, value):
        self._set_skew_entries(value)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_diag_entries(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_diag_entries)
        self.initialize(raw_diag_entries=self.raw_diag_entries_constraint.inverse_transform(value))

    def _set_skew_entries(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_skew_entries)
        self.initialize(raw_skew_entries=self.raw_skew_entries_constraint.inverse_transform(value))

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    @property
    def skew_matrix(self):
        S = torch.zeros((self.num_outputs, self.num_inputs, self.num_inputs), dtype=torch.float64)
        if self.n_skew_entries > 0:
            # to construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
            S[self.skew_matrix_indices] = self.skew_entries
            S += -S.transpose(-1, -2)
        return torch.linalg.matrix_exp(S)

    @property
    def diag_matrix(self):
        D = torch.zeros((self.num_outputs, self.num_inputs, self.num_inputs), dtype=torch.float64)
        D[self.diag_matrix_indices] = self.diag_entries.ravel()
        return D

    @property
    def latent_transform(self):
        return torch.matmul(self.diag_matrix, self.skew_matrix)

    def forward(self, x1, x2, diag=False, **params):
        # adapted from the Matern kernel
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

        trans_x1 = torch.matmul(self.latent_transform.unsqueeze(1), (x1 - mean).unsqueeze(-1)).squeeze(-1)
        trans_x2 = torch.matmul(self.latent_transform.unsqueeze(1), (x2 - mean).unsqueeze(-1)).squeeze(-1)

        distance = self.covar_dist(trans_x1, trans_x2, diag=diag, **params)

        if self.num_outputs == 1:
            distance = distance.squeeze(0)

        outputscale = self.outputscale if self.scale_output else 1.0

        # special cases of the Matern function
        if self.nu == 0.5:
            return outputscale * torch.exp(-distance)
        if self.nu == 1.5:
            return outputscale * (1 + distance) * torch.exp(-distance)
        if self.nu == 2.5:
            return outputscale * (1 + distance + distance**2 / 3) * torch.exp(-distance)
