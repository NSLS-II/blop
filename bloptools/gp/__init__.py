import time as ttime
import warnings

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401
import gpytorch
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import torch
from matplotlib import pyplot as plt
from scipy.stats import qmc

from . import kernels, plans, utils


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

    def set_data(self, x, y):
        """
        Set the data with parameters and values.

        x: parameters
        y: function values at those parameters
        """

        if np.isnan(y).any():
            raise ValueError("One of the passed values is NaN.")

        self.x, self.y = np.atleast_2d(x), np.atleast_1d(y)
        self.n, self.n_dof = self.x.shape

        # prepare Gaussian process ingredients for the regressor and classifier
        # use only regressable points for the regressor
        self.inputs = torch.as_tensor(self.x).float()
        self.targets = torch.as_tensor(self.y).float()

        self.noise_upper_bound = 1e-1 * self.max_noise_fraction if len(self.y) > 1 else self.max_noise_fraction
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(0, self.noise_upper_bound)
        )

        self.model = GaussianProcessModel(
            self.inputs,
            self.targets,
            self.likelihood,
            self.n_dof,
            self.length_scale_bounds,
        )

        self.init_state_dict = self.model.state_dict()
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)

    def train(self, training_iter=100, reuse_hypers=True, verbose=True):
        if not reuse_hypers:
            self.model.load_state_dict(self.init_state_dict)

        self.likelihood.train()
        self.model.train()

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            self.optimizer.zero_grad()
            loss = -self.mll(self.model(self.inputs), self.targets)
            loss.backward()
            self.optimizer.step()

            if verbose and ((i + 1) % 100 == 0):
                print(f"{i+1}/{training_iter} inverse_length_scales: {self.model.covar_module.trans_diagonal}")

        self.state_dict = self.model.state_dict()

    def regress(self, x):
        x = torch.as_tensor(np.atleast_2d(x)).float()

        # set to evaluation mode
        self.likelihood.eval()
        self.model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.model(x))

        return prediction

    def mean(self, x):
        return self.regress(x).mean.detach().numpy().ravel()

    def sigma(self, x):
        return self.regress(x).stddev.detach().numpy().ravel()

    @property
    def nu(self):
        return self.y.max()


class GPC:
    """
    A Gaussian process classifier, with learning methods.
    """

    def __init__(self, length_scale_bounds=(1e-3, 1e0), **kwargs):
        self.state_dict = None
        self.length_scale_bounds = length_scale_bounds

    def set_data(self, x, y):
        """
        Set the data with parameters and values.

        x: parameters
        y: function values at those parameters

        Passed parameters must be between [-1, 1] in every dimension. Passed values must be integer labels.
        """

        # if (x.min(axis=0) <= -1).any() or (x.max(axis=0) >= +1).any():
        #    raise ValueError('Parameters must be between -1 and +1 in each dimension.')

        self.x, self.y = np.atleast_2d(x), np.atleast_1d(y).astype(int)
        self.n, self.n_dof = self.x.shape

        self.inputs = torch.as_tensor(self.x).float()
        self.targets = torch.as_tensor(self.y)

        self.likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            self.targets, learn_additional_noise=True
        )

        self.model = GaussianProcessModel(
            self.inputs,
            self.likelihood.transformed_targets,
            self.likelihood,
            self.n_dof,
            self.length_scale_bounds,
            batch_shape=2,
        )

        self.init_state_dict = self.model.state_dict()
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)

    def train(self, training_iter=100, reuse_hypers=True, verbose=True):
        if not reuse_hypers:
            self.model.load_state_dict(self.init_state_dict)

        self.likelihood.train()
        self.model.train()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-1
        )  # Includes GaussianLikelihood parameters
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            self.optimizer.zero_grad()
            loss = -self.mll(self.model(self.inputs), self.likelihood.transformed_targets).sum()
            loss.backward()
            self.optimizer.step()

            if verbose and ((i + 1) % 100 == 0):
                print(f"{i+1}/{training_iter} inverse_length_scales: {self.model.covar_module.trans_diagonal}")

        self.state_dict = self.model.state_dict()

    def classify(self, x, return_variance=False):
        x = torch.as_tensor(np.atleast_2d(x)).float()

        # set to evaluation mode
        self.likelihood.eval()
        self.model.eval()

        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            dist = self.model(x)
            samples = dist.sample(torch.Size((256,))).exp()
            probabilities = (samples / samples.sum(-2, keepdim=True)).mean(0)

        if return_variance:
            res = probabilities[1].detach().numpy(), dist.variance.detach().numpy()
        else:
            res = probabilities[1].detach().numpy()

        return res


def load(filepath, **kwargs):
    with h5py.File(filepath, "r") as f:
        params = f["params"][:]
    data = pd.read_hdf(filepath, key="data")
    return Optimizer(init_params=params, init_data=data, **kwargs)


class Optimizer:
    def __init__(
        self,
        detector,
        detector_type,  # either "image" or "scalar"
        dofs,
        dof_bounds,
        run_engine,
        db,
        fitness_model,
        shutter=None,
        init_params=None,
        init_data=None,
        init_scheme=None,
        n_init=None,
        training_iter=256,
        verbose=True,
        **kwargs,
    ):
        self.dofs, self.dof_bounds = dofs, dof_bounds
        self.n_dof = len(dofs)

        self.detector_type = detector_type

        self.fitness_model = fitness_model

        self.run_engine = run_engine
        self.detector = detector

        self.db = db
        self.training_iter = training_iter

        self.gp_lp_fig = None
        self.fit_lp_fig = None

        # convert params to x
        self.params_trans_fun = (
            lambda params: 2 * (params - self.dof_bounds.min(axis=1)) / self.dof_bounds.ptp(axis=1) - 1
        )

        # convert x to params
        self.inv_params_trans_fun = lambda x: 0.5 * (x + 1) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(
            axis=1
        )

        self.shutter = shutter

        self.fig, self.axes = None, None

        if self.shutter is not None:
            (uid,) = self.run_engine(plans.take_background(self))
            self.background = np.array(list(db[uid].data(field=f"{self.detector.name}_image"))[0])

            if self.shutter.status.get() != 0:
                raise RuntimeError("Could not open shutter!")

        else:
            self.background = 0

        # for actual prediction and optimization
        self.evaluator = GPR(
            length_scale_bounds=(1e-1, 1e0), max_noise_fraction=1e-2
        )  # at most 1% of the RMS is due to noise
        self.timer = GPR(length_scale_bounds=(5e-1, 2e0), max_noise_fraction=1e0)  # can be noisy, why not
        self.validator = GPC(length_scale_bounds=(1e-1, 1e0))

        self.params = np.zeros((0, self.n_dof))
        self.data = pd.DataFrame()

        if (init_params is not None) and (init_data is not None):
            pass

        elif init_scheme == "quasi-random":
            n_init = n_init if n_init is not None else 3**self.n_dof
            init_params, init_data = self.autoinitialize(n=n_init, scheme="quasi-random", verbose=verbose)

        else:
            raise Exception(
                "Could not initialize model! Either pass initial params and data, or specify one of:"
                "['quasi-random']."
            )

        self.append(new_params=init_params, new_data=init_data)
        self.update(reuse_hypers=True, verbose=verbose)  # update our model

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.test_params = sampler.random(n=2048) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(axis=1)
        self.test_params = self.test_params[self.test_params[:, 0].argsort()]

    @property
    def current_params(self):
        return np.array([dof.user_readback.get() for dof in self.dofs])

    @property
    def optimum(self):
        return self.inv_params_trans_fun(self.evaluator.x[np.nanargmax(self.evaluator.y)])

    def go_to_optimum(self):
        self.run_engine(bps.mv(*[_ for items in zip(self.dofs, np.atleast_1d(self.optimum).T) for _ in items]))

    def inspect_beam(self, index, masked=True):
        im = self.masked_images[index] if masked else self.images[index]
        plt.figure()
        plt.imshow(im)

        bbx = self.parsed_image_data.loc[index, ["x_min", "x_max"]].values[[0, 0, 1, 1, 0]]
        bby = self.parsed_image_data.loc[index, ["y_min", "y_max"]].values[[0, 1, 1, 0, 0]]

        plt.plot(bbx, bby, lw=1e0, c="r")

    def save(self, filepath="./model.gpo"):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("params", data=self.params)
        self.data.to_hdf(filepath, key="data")

    def compute_fitness(self):
        if self.detector_type == "image":
            if f"{self.detector.name}_vertical_extent" in self.data.columns:
                x_extents = list(
                    map(
                        np.array,
                        [
                            e if len(np.atleast_1d(e)) == 2 else (0, 1)
                            for e in self.data[f"{self.detector.name}_vertical_extent"]
                        ],
                    )
                )
                y_extents = list(
                    map(
                        np.array,
                        [
                            e if len(np.atleast_1d(e)) == 2 else (0, 1)
                            for e in self.data[f"{self.detector.name}_horizontal_extent"]
                        ],
                    )
                )
                extents = np.c_[y_extents, x_extents]

            else:
                extents = None

            self.images = (
                np.r_[[image for image in self.data[f"{self.detector.name}_image"].values]] - self.background
            )

            mask = (self.images > 0.05 * self.images.max(axis=(-1, -2))[:, None, None]).astype(int)

            self.masked_images = mask * self.images

            self.parsed_image_data = utils.parse_images(self.masked_images, extents, remove_background=False)

            self.fitness = self.parsed_image_data.fitness.values

            # convert fitness to y
            self.fitness_trans_fun = lambda fitness: np.log(fitness)

            # convert y to fitness
            self.inv_fitness_trans_fun = lambda y: np.exp(y)

        # if self.detector_type == 'scalar':

        #    self.fitness =

    def draw_fitness_liveplot(self):
        if self.fit_lp_fig is None:
            self.fit_lp_fig, self.fit_lp_axes = plt.subplots(
                1, 2, figsize=(8, 4), dpi=128, sharex=True, sharey=True, constrained_layout=True
            )

        self.fit_lp_axes[0].clear()

        times = self.data.time.astype(int).values / 1e9
        times -= times[0]

        cum_max_fitness = [np.nanmax(self.fitness[: i + 1]) for i in range(len(self.fitness))]

        self.fit_lp_axes[0].scatter(times, self.fitness, c="k")
        self.fit_lp_axes[0].plot(times, cum_max_fitness, c="r")
        self.fit_lp_axes[0].set_yscale("log")

        self.fit_lp_fig.canvas.draw_idle()
        self.fit_lp_fig.show()

    def draw_gp_liveplot(self):
        if self.gp_lp_fig is None:
            self.gp_lp_fig, self.gp_lp_axes = plt.subplots(
                2, 3, figsize=(12, 8), dpi=128, sharex=True, sharey=True, constrained_layout=True
            )

        size = 64
        norm = mpl.colors.LogNorm(*np.nanpercentile(self.fitness, q=[1, 99]))

        from matplotlib.patches import Patch

        exploit_params = self.recommend(strategy="exploit", n=1)
        explore_params = self.recommend(strategy="explore", n=1)

        p_valid = self.validate(self.test_params)

        self.gp_lp_axes[0, 0].clear()
        self.gp_lp_axes[0, 0].scatter(*self.params.T[:2], s=size, c=self.fitness, norm=norm, cmap="coolwarm")
        self.gp_lp_axes[0, 0].set_title("sampled fitness")

        self.gp_lp_axes[1, 0].clear()
        self.gp_lp_axes[1, 0].scatter(*self.params.T[:2], s=size, c=self.c, vmin=0, vmax=1, cmap="viridis")
        self.gp_lp_axes[1, 0].set_title("sampled goodness")
        handles = [
            Patch(label="good", color=plt.cm.viridis.colors[-1]),
            Patch(label="bad", color=plt.cm.viridis.colors[0]),
        ]
        self.gp_lp_axes[1, 0].legend(handles=handles)

        self.gp_lp_axes[0, 1].clear()
        self.gp_lp_axes[0, 1].scatter(
            *self.test_params.T[:2], s=size, c=self.fitness_estimate(self.test_params), norm=norm, cmap="coolwarm"
        )
        self.gp_lp_axes[0, 1].scatter(
            *exploit_params.T[:2], marker="*", facecolor="w", edgecolor="k", s=4 * size, label="max_improvement"
        )
        self.gp_lp_axes[0, 1].set_title("fitness estimate")

        self.gp_lp_axes[1, 1].clear()
        self.gp_lp_axes[1, 1].scatter(*self.test_params.T[:2], s=size, c=p_valid, vmin=0, vmax=1, cmap="viridis")
        self.gp_lp_axes[1, 1].set_title("goodness estimate")

        self.gp_lp_axes[0, 2].clear()
        self.gp_lp_axes[0, 2].scatter(
            *self.test_params.T[:2], s=size, c=self.fitness_entropy(self.test_params), cmap="coolwarm"
        )
        self.gp_lp_axes[0, 2].scatter(
            *explore_params.T[:2], marker="s", facecolor="w", edgecolor="k", s=4 * size, label="max_information"
        )
        self.gp_lp_axes[0, 2].set_title("fitness entropy")

        self.gp_lp_axes[1, 2].clear()
        self.gp_lp_axes[1, 2].scatter(
            *self.test_params.T[:2], s=size, c=np.log((1 - p_valid) * p_valid), cmap="viridis"
        )
        self.gp_lp_axes[1, 2].set_title("goodness entropy")

        for ax in self.gp_lp_axes.ravel():
            ax.set_xlim(*self.dof_bounds[0])
            ax.set_ylim(*self.dof_bounds[1])

        self.gp_lp_fig.canvas.draw_idle()
        self.gp_lp_fig.show()

    def autoinitialize(self, n, verbose, scheme="quasi-random"):
        halton_sampler = qmc.Halton(d=self.n_dof, scramble=True)

        params_to_sample = halton_sampler.random(n=2 ** int(np.log(n) / np.log(2) + 1))[:n] * self.dof_bounds.ptp(
            axis=1
        ) + self.dof_bounds.min(axis=1)
        # self.params = np.r_[self.dof_bounds.mean(axis=0)[None], self.params]
        sampled_params, res_table = self.acquire_with_bluesky(params_to_sample, verbose=verbose)

        return sampled_params, res_table

    def append(self, new_params, new_data):
        self.params = np.r_[self.params, new_params]
        self.data = pd.concat([self.data, new_data])

    def update(self, reuse_hypers=True, verbose=False):
        self.compute_fitness()

        self.x = self.params_trans_fun(self.params)
        self.y = self.fitness_trans_fun(self.fitness)
        self.c = (~np.isnan(self.y)).astype(int)

        self.evaluator.set_data(self.x[self.c == 1], self.y[self.c == 1])
        self.validator.set_data(self.x, self.c)
        self.timer.set_data(np.abs(np.diff(self.x, axis=0)), self.data.acq_duration.values[1:])

        self.timer.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.evaluator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.validator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)

    def acquire(self, params):
        pass

    def acquire_with_bluesky(self, params, routing=True, verbose=False):
        if routing:
            routing_index, _ = utils.get_routing(self.current_params, params)
            ordered_params = params[routing_index]

        else:
            ordered_params = params

        table = pd.DataFrame(columns=["acq_time", "acq_duration", "acq_log"])

        for _params in ordered_params:
            if verbose:
                print(f"sampling {_params}")

            start_params = self.current_params
            # rel_d_params = (_params - start_params) / self.dof_bounds.ptp(axis=1)

            # acq_delay = utils.get_movement_time(rel_d_params, v_max=0.25, a=0.5).max()
            # print(f'delay: {acq_delay}')

            start_time = ttime.monotonic()
            # ttime.sleep(acq_delay)

            try:
                (uid,) = self.run_engine(
                    bp.list_scan(
                        [self.detector], *[_ for items in zip(self.dofs, np.atleast_2d(_params).T) for _ in items]
                    )
                )
                _table = self.db[uid].table(fill=True)
                _table.insert(0, "acq_time", ttime.time())
                _table.insert(1, "acq_duration", ttime.monotonic() - start_time)
                _table.insert(2, "acq_log", "ok")
                _table.insert(3, "uid", uid)

            except Exception as err:
                warnings.warn(err.args[0])
                columns = ["acq_time", "acq_duration", "acq_log", "uid", f"{self.detector.name}_image"]
                _table = pd.DataFrame(
                    [
                        (
                            ttime.time(),
                            ttime.monotonic() - start_time,
                            err.args[0],
                            "",
                            np.zeros(self.detector.shape.get()),
                        )
                    ],
                    columns=columns,
                )

            for start_param, dof in zip(start_params, self.dofs):
                _table.loc[:, f"delta_{dof.name}"] = dof.user_readback.get() - start_param

            table = pd.concat([table, _table])

        return ordered_params, table

    def recommend(self, strategy=None, greedy=True, rate=False, n=1, n_test=256):
        """
        Recommends the next $n$ points to sample, according to the given strategy.
        """

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)

        # recommend some parameters that we might want to sample, with shape (., n, n_dof)
        TEST_PARAMS = (
            sampler.random(n=n * n_test) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(axis=1)
        ).reshape(n_test, n, self.n_dof)

        # how much will we have to change our parameters to sample these guys?
        DELTA_TEST_PARAMS = np.diff(
            np.concatenate([np.repeat(self.current_params[None, None], n_test, axis=0), TEST_PARAMS], axis=1),
            axis=1,
        )

        # how long will that take?
        if rate:
            expected_total_delay = self.delay_estimate(DELTA_TEST_PARAMS).sum(axis=1)
            if not np.all(expected_total_delay > 0):
                raise ValueError("Some estimated acquisition times are non-positive.")

        if greedy:
            if strategy.lower() == "exploit":  # greedy expected reward maximization
                objective = -self._negative_expected_improvement(TEST_PARAMS).sum(axis=1)

        if strategy.lower() == "explore":
            objective = -self._negative_expected_information_gain(TEST_PARAMS)

        if rate:
            objective /= expected_total_delay
        return TEST_PARAMS[np.argmax(objective)]

    def learn(
        self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
    ):
        # ip.display.clear_output(wait=True)
        print(f'learning with strategy "{strategy}" ...')

        for i in range(n_iter):
            params_to_sample = np.atleast_2d(
                self.recommend(n=n_per_iter, strategy=strategy, **kwargs)
            )  # get point(s) to sample from the strategizer

            n_original = len(params_to_sample)
            n_upsample = upsample * (n_original - 1) + 1

            upsampled_params_to_sample = sp.interpolate.interp1d(np.arange(n_original), params_to_sample.T)(
                np.linspace(0, n_original - 1, n_upsample)
            ).T

            sampled_params, res_table = self.acquire_with_bluesky(
                upsampled_params_to_sample
            )  # sample the point(s)
            self.append(new_params=sampled_params, new_data=res_table)
            self.update(reuse_hypers=reuse_hypers)  # update our model

            if "gp" in plots:
                self.draw_gp_liveplot()
            if "fitness" in plots:
                self.draw_fitness_liveplot()

            if verbose:
                # self.plot_readback()
                print(f"# {i+1:>03} : {params_to_sample.round(4)} -> {self.fitness[-1]:.04e}")

    def plot_state(self):
        # cm = mpl.cm.get_cmap("coolwarm")

        fig, axes = mpl.pyplot.subplots(2, 4, figsize=(12, 8), dpi=128, sharex=True, sharey=True)

        # evaluated over test_params
        # current_info = -self._posterior_entropy(params=None)
        # potential_info = -self._posterior_entropy(params=self.test_params[:, None, :])

        p_valid = self.validate(self.test_params)

        norm = mpl.colors.LogNorm(*np.nanpercentile(self.fitness, q=[1, 99]))

        s = 32

        # plot values of data points
        ax = axes[0, 0]
        ax.set_title("fitness")
        ref = ax.scatter(*self.params.T[:2], s=s, c=self.fitness, norm=norm)
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        # axes[0,0].scatter(*max_prior_entropy_params[:2], marker='o', color='k', s=s, label='max_prior_entropy')
        max_improvement_params = self.recommend(strategy="exploit", n=1)
        max_information_params = self.recommend(strategy="explore", n=1)
        axes[0, 0].scatter(*max_information_params.T[:2], marker="s", color="k", s=s, label="max_information")
        axes[0, 0].scatter(*max_improvement_params.T[:2], marker="*", color="k", s=s, label="max_improvement")
        axes[0, 0].legend(fontsize=6)

        # plot the estimate of test points
        ax = axes[0, 1]
        ax.set_title("fitness estimate")
        ref = ax.scatter(*self.test_params.T[:2], s=s, c=self.fitness_estimate(self.test_params), norm=norm)
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        # plot the entropy rate of test points
        ax = axes[0, 2]
        ax.set_title("fitness entropy")
        ref = ax.scatter(
            *self.test_params.T[:2], s=s, c=self.fitness_entropy(self.test_params), norm=mpl.colors.LogNorm()
        )
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        # plot the estimate of test points
        ax = axes[0, 3]
        ax.set_title("delay estimate")
        ref = ax.scatter(*self.test_params.T[:2], s=s, c=self.delay_estimate(self.test_params))
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        # plot classification of data points
        ax = axes[1, 0]
        ax.set_title("validity")
        ref = ax.scatter(*self.params.T[:2], s=s, c=self.c, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        ax = axes[1, 1]
        ax.set_title("validity estimate")
        ref = ax.scatter(*self.test_params.T[:2], s=s, c=p_valid, vmin=0, vmax=1)
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        ax = axes[1, 2]
        ax.set_title("greedy improvement")
        ref = ax.scatter(
            *self.test_params.T[:2],
            s=s,
            c=-self._negative_expected_improvement(self.test_params) / self.delay_estimate(self.test_params),
        )
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        ax = axes[1, 3]
        ax.set_title("greedy information")
        ref = ax.scatter(
            *self.test_params.T[:2],
            s=s,
            c=-self._negative_expected_information_gain(self.test_params[:, None, :])
            / self.delay_estimate(self.test_params),
        )
        fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

    def plot_readback(self):
        # cm = mpl.cm.get_cmap("coolwarm")

        p_valid = self.validate(self.test_params)
        norm = mpl.colors.LogNorm(*np.nanpercentile(self.fitness, q=[1, 99]))
        s = 32

        if self.fig is None:
            self.fig, self.axes = mpl.pyplot.subplots(2, 2, figsize=(12, 8), dpi=128, sharex=True, sharey=True)

        # plot values of data points
        ax = self.fig.axes[0]
        ax.clear()
        ax.set_title("fitness")
        ref = ax.scatter(*self.params.T[:2], s=s, c=self.fitness, norm=norm)
        self.fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        # plot the estimate of test points
        ax = self.fig.axes[1]
        ax.clear()
        ax.set_title("fitness estimate")
        ref = ax.scatter(*self.test_params.T[:2], s=s, c=self.mean(self.test_params), norm=norm)
        self.fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        # plot classification of data points
        ax = self.fig.axes[2]
        ax.clear()
        ax.set_title("class")
        ref = ax.scatter(*self.params.T[:2], s=s, c=self.c, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        self.fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

        ax = self.fig.axes[3]
        ax.clear()
        ax.set_title("class estimate")
        ref = ax.scatter(*self.test_params.T[:2], s=s, c=1 - p_valid, vmin=0, vmax=1)
        self.fig.colorbar(ref, ax=ax, location="bottom", aspect=32)

    def _negative_improvement_variance(self, params):
        x = self.params_trans_fun(params)

        mu = self.evaluator.mean(x)
        sigma = self.evaluator.sigma(x)
        nu = self.evaluator.nu
        p = self.validator.classify(x)

        # sigma += 1e-3 * np.random.uniform(size=sigma.shape)

        A = np.exp(-0.5 * np.square((mu - nu) / sigma)) / (np.sqrt(2 * np.pi) * sigma)
        B = 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))

        V = -(p**2) * (A * sigma**2 + B * (mu - nu)) ** 2 + p * (
            A * sigma**2 * (mu - nu) + B * (sigma**2 + (mu - nu) ** 2)
        )

        return -np.maximum(0, V)

    # talk to the model

    def fitness_estimate(self, params):
        return self.inv_fitness_trans_fun(
            self.evaluator.mean(self.params_trans_fun(params).reshape(-1, self.n_dof))
        ).reshape(params.shape[:-1])

    def fitness_sigma(self, params):
        return self.inv_fitness_trans_fun(
            self.evaluator.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof))
        ).reshape(params.shape[:-1])

    def fitness_entropy(self, params):
        return np.log(np.sqrt(2 * np.pi * np.e) * self.fitness_sigma(params) + 1e-12)

    def validate(self, params):
        return self.validator.classify(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
            params.shape[:-1]
        )

    def delay_estimate(self, params):
        return self.timer.mean(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

    def delay_sigma(self, params):
        return self.timer.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

    def _negative_expected_improvement(self, params):
        """
        Returns the negative expected improvement over the maximum, in GP units.
        """

        x = self.params_trans_fun(params).reshape(-1, self.n_dof)

        # using GPRC units here
        mu = self.evaluator.mean(x)
        sigma = self.evaluator.sigma(x)
        nu = self.evaluator.nu
        p = self.validator.classify(x)

        A = np.exp(-0.5 * np.square((mu - nu) / sigma)) / (np.sqrt(2 * np.pi) * sigma)
        B = 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))
        E = -p * (A * sigma**2 + B * (mu - nu))

        return E.reshape(params.shape[:-1])

    # functions for expected_information strategies

    def _negative_expected_information_gain(self, params):
        current_info = -self._posterior_entropy(params=None)
        potential_info = -self._posterior_entropy(params=params)
        p_valid = self.validate(params)

        n_bad, n_tot = (potential_info - current_info <= 0).sum(), len(potential_info.ravel())

        if not n_bad == 0:  # the posterior variance should always be positive
            warnings.warn(f"{n_bad}/{n_tot} information estimates are non-positive.")
            if n_bad / n_tot > 0.5:
                raise ValueError("More than half of the information estimates are non-positive.")

        return -np.product(p_valid, axis=-1) * (potential_info - current_info)

    def _posterior_entropy(self, params=None):
        """
        params is an array with shape (n_sets, n_params_per_set, n_dof)

        If we observe each of the n_params_per_set in each of the n_sets, what will the resulting integrals
        over the posterior rate be? This function estimates that using a Quasi-Monte Carlo integration over a
        dummy Gaussian processes.
        Returns an array of shape (n_sets,).

        If None is passed, we return the posterior entropy of the real process.
        """

        if params is None:
            params = np.empty((1, 0, self.n_dof))  # one set of zero observations

        # get the noise from the evaluator likelihood
        raw_noise = self.evaluator.model.state_dict()["likelihood.noise_covar.raw_noise"]
        noise = self.evaluator.model.likelihood.noise_covar.raw_noise_constraint.transform(raw_noise).item()

        # n_data is the number of points in each observation we consider (n_data = n_process + n_params_per_set)
        # x_data is an array of shape (n_sets, n_data, n_dof) that describes potential obervation states
        # x_star is an array of points at which to evaluate the entropy rate, to sum together for the QMCI
        x_data = torch.as_tensor(
            np.r_[[np.r_[self.evaluator.x, _x] for _x in np.atleast_3d(self.params_trans_fun(params))]]
        )
        x_star = torch.as_tensor(self.params_trans_fun(self.test_params))

        # for each potential observation state, compute the prior-prior and prior-posterior covariance matrices
        # $C_data_data$ is the covariance of the potential data with itself, for each set of obervations
        # $C_star_data$ is the covariance of the QMCI points with the potential data (n_sets, n_qmci, n_data)
        # we don't care about K_star_star for our purposes, only its diagonal which is a constant prior_variance
        C_data_data = self.evaluator.model.covar_module(x_data, x_data).detach().numpy().astype(
            float
        ) + noise**2 * np.eye(x_data.shape[1])
        C_star_data = self.evaluator.model.covar_module(x_star, x_data).detach().numpy().astype(float)

        prior_variance = self.evaluator.model.covar_module.output_scale.item() ** 2 + noise**2

        # normally we would compute A * B" * A', but that would be inefficient as we only care about the diagonal.
        # instead, compute this as:
        #
        # diag(A * B" * A') = sum(A * B" . A', -1)
        #
        # which is much faster.

        explained_variance = (np.matmul(C_star_data, np.linalg.inv(C_data_data)) * C_star_data).sum(axis=-1)
        posterior_variance = prior_variance - explained_variance

        n_bad, n_tot = (posterior_variance <= 0).sum(), len(posterior_variance.ravel())

        if not n_bad == 0:  # the posterior variance should always be positive
            warnings.warn(f"{n_bad}/{n_tot} variance estimates are non-positive.")
            if n_bad / n_tot > 0.5:
                raise ValueError("More than half of the variance estimates are non-positive.")

        marginal_entropy_rate = 0.5 * np.log(2 * np.pi * np.e * posterior_variance)

        return marginal_entropy_rate.sum(axis=-1)
