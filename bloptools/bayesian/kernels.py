import gpytorch
import numpy as np
import torch


class LatentKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    num_outputs = 1

    def __init__(
        self,
        num_inputs=1,
        off_diag=True,
        diag_prior=True,
        scale_kernel=True,
        **kwargs,
    ):
        super(LatentKernel, self).__init__()

        self.num_inputs = num_inputs
        self.n_off_diag = int(num_inputs * (num_inputs - 1) / 2)

        self.off_diag = off_diag
        self.scale_kernel = scale_kernel

        self.nu = kwargs.get("nu", 1.5)

        # kernel_scale_constraint = gpytorch.constraints.Positive()
        diag_entries_constraint = gpytorch.constraints.Positive()  # gpytorch.constraints.Interval(5e-1, 1e2)
        skew_entries_constraint = gpytorch.constraints.Interval(-1e0, 1e0)

        # diag_entries_initial = np.ones()
        # np.sqrt(diag_entries_constraint.lower_bound * diag_entries_constraint.upper_bound)
        raw_diag_entries_initial = diag_entries_constraint.inverse_transform(torch.tensor(2))

        self.register_parameter(
            name="raw_diag_entries",
            parameter=torch.nn.Parameter(raw_diag_entries_initial * torch.ones(self.num_outputs, self.num_inputs).double()),
        )
        self.register_constraint("raw_diag_entries", constraint=diag_entries_constraint)

        if diag_prior:
            self.register_prior(
                name="diag_entries_prior",
                prior=gpytorch.priors.GammaPrior(concentration=2, rate=1),
                param_or_closure=lambda m: m.diag_entries,
                setting_closure=lambda m, v: m._set_diag_entries(v),
            )

        if self.off_diag:
            self.register_parameter(
                name="raw_skew_entries",
                parameter=torch.nn.Parameter(torch.zeros(self.num_outputs, self.n_off_diag).double()),
            )
            self.register_constraint("raw_skew_entries", skew_entries_constraint)

        if self.scale_kernel:
            kernel_scale_constraint = gpytorch.constraints.Positive()
            kernel_scale_prior = gpytorch.priors.GammaPrior(concentration=2, rate=0.15)

            self.register_parameter(
                name="raw_kernel_scale",
                parameter=torch.nn.Parameter(torch.ones(1)),
            )

            self.register_constraint("raw_kernel_scale", constraint=kernel_scale_constraint)

            self.register_prior(
                name="kernel_scale_prior",
                prior=kernel_scale_prior,
                param_or_closure=lambda m: m.kernel_scale,
                setting_closure=lambda m, v: m._set_kernel_scale(v),
            )

    @property
    def diag_entries(self):
        return self.raw_diag_entries_constraint.transform(self.raw_diag_entries)

    @property
    def skew_entries(self):
        return self.raw_skew_entries_constraint.transform(self.raw_skew_entries)

    @property
    def kernel_scale(self):
        return self.raw_kernel_scale_constraint.transform(self.raw_kernel_scale)

    @diag_entries.setter
    def diag_entries(self, value):
        self._set_diag_entries(value)

    @skew_entries.setter
    def skew_entries(self, value):
        self._set_skew_entries(value)

    @kernel_scale.setter
    def kernel_scale(self, value):
        self._set_kernel_scale(value)

    def _set_diag_entries(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_diag_entries)
        self.initialize(raw_diag_entries=self.raw_diag_entries_constraint.inverse_transform(value))

    def _set_skew_entries(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_skew_entries)
        self.initialize(raw_skew_entries=self.raw_skew_entries_constraint.inverse_transform(value))

    def _set_kernel_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kernel_scale)
        self.initialize(raw_kernel_scale=self.raw_kernel_scale_constraint.inverse_transform(value))

    @property
    def output_scale(self):
        return self.kernel_scale.sqrt()

    @property
    def latent_dimensions(self):
        # no rotations
        if not self.off_diag:
            T = torch.eye(self.num_inputs, dtype=torch.float64)

        # construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
        else:
            A = torch.zeros((self.num_inputs, self.num_inputs)).double()
            A[np.triu_indices(self.num_inputs, k=1)] = self.skew_entries
            A += -A.transpose(-1, -2)
            T = torch.linalg.matrix_exp(A)

        diagonal_transform = torch.cat([torch.diag(entries).unsqueeze(0) for entries in self.diag_entries], dim=0)
        T = torch.matmul(diagonal_transform, T)

        return T

    def forward(self, x1, x2, diag=False, **params):
        # adapted from the Matern kernel
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

        trans_x1 = torch.matmul(self.latent_dimensions.unsqueeze(1), (x1 - mean).unsqueeze(-1)).squeeze(-1)
        trans_x2 = torch.matmul(self.latent_dimensions.unsqueeze(1), (x2 - mean).unsqueeze(-1)).squeeze(-1)

        distance = self.covar_dist(trans_x1, trans_x2, diag=diag, **params)

        return self.kernel_scale * (1 + distance) * torch.exp(-distance)
