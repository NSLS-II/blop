import time as ttime
import warnings

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import torch
from matplotlib import pyplot as plt

from . import configs, fitness, models, plans, utils

mpl.rc("image", cmap="coolwarm")

print('reloaded!')


def _negative_expected_information_gain(evaluator, validator, base_X, experimental_X):
    qmc_X = sp.stats.qmc.Halton(d=base_X.shape[-1], scramble=True).random(n=1024) * 2 - 1

    potential_info = -_posterior_entropy(evaluator, base_X, experimental_X, qmc_X=qmc_X)
    current_info = -_posterior_entropy(evaluator, base_X, experimental_X=None, qmc_X=qmc_X)

    PV = validator.classify(experimental_X.reshape(-1, evaluator.n_dof)).reshape(experimental_X.shape[:-1])

    n_bad, n_tot = (potential_info - current_info <= 0).sum(), len(potential_info.ravel())

    if not n_bad == 0:  # the posterior variance should always be positive
        warnings.warn(f"{n_bad}/{n_tot} information estimates are non-positive.")
        if n_bad / n_tot > 0.5:
            raise ValueError(
                f"More than half ({int(1e2*n_bad/n_tot)}%) of the information estimates are non-positive."
            )

    return -np.product(PV, axis=-1) * (potential_info - current_info)


def _posterior_entropy(evaluator, base_X, experimental_X=None, qmc_X=None):
    """
    Returns the change in integrated entropy of the process contingent on some experiment using a Quasi-
    Monte Carlo integration over a dummy Gaussian processes.

    prior_params: numpy array with shape (n_params, n_dof)

    experimental_X: numpy array with shape (n_sets, n_params_per_set, n_dof)

    If None is passed, we return the posterior entropy of the real process.
    """

    if qmc_X is None:
        qmc_X = sp.stats.qmc.Halton(d=base_X.shape[-1], scramble=True).random(n=1024) * 2 - 1

    if experimental_X is None:
        experimental_X = np.empty((1, 0, base_X.shape[-1]))  # one set of zero observations

    if not experimental_X.ndim == 3:
        raise ValueError("Passed params must have shape (n_sets, n_params_per_set, n_dof).")

    # get the noise from the evaluator likelihood
    noise = float(evaluator.model.likelihood.noise.item())

    b_X = torch.as_tensor(base_X)  # our base points that we've already sampled
    e_X = torch.as_tensor(experimental_X)  # points that we are thinking of sampling
    q_X = torch.as_tensor(qmc_X)  # points that we want our experiment to explain

    n_b = b_X.shape[0]  # how many base points we have
    n_s = e_X.shape[0]  # how many set of experiments we're computing this stuff for
    n_e = e_X.shape[1]  # how many points are in each set of experiments

    C_pp = evaluator.model.covar_module(b_X, b_X).detach().numpy().astype(float) + noise * np.eye(n_b)

    # compute the ingredients of the factors of the blockwise inverse
    A = np.repeat(C_pp[None], n_s, axis=0)
    B = evaluator.model.covar_module(b_X, e_X).detach().numpy().astype(float)
    C = np.swapaxes(B, -1, -2)
    D = evaluator.model.covar_module(e_X, e_X).detach().numpy().astype(float) + noise * np.eye(n_e)[None]

    invA = np.repeat(np.linalg.inv(C_pp)[None], n_s, axis=0)
    invD = np.linalg.inv(D)

    # compute the two factors of the blockwise inverse
    BWI_1 = np.zeros((len(e_X), n_b + n_e, n_b + n_e))
    BWI_1[:, :n_b, :n_b] = np.linalg.inv(A - utils.mprod(B, invD, C))
    BWI_1[:, n_b:, n_b:] = np.linalg.inv(D - utils.mprod(C, invA, B))

    BWI_2 = np.repeat(np.eye(n_b + n_e)[None], n_s, axis=0)
    BWI_2[:, :n_b, n_b:] = -utils.mprod(B, invD)
    BWI_2[:, n_b:, :n_b] = -utils.mprod(C, invA)

    # we can now use BWI_1 * BWI_2 instead of [[A, B], [C, D]]"

    # the covariance of the base and experiment points with the QMC points
    C_qb = np.repeat(evaluator.model.covar_module(q_X, b_X).detach().numpy().astype(float)[None], n_s, axis=0)
    C_qe = evaluator.model.covar_module(q_X, e_X).detach().numpy().astype(float)
    C_qt = np.c_[C_qb, C_qe]

    prior_variance = evaluator.model.covar_module.output_scale.item() ** 2 + noise

    # normally we would compute the posterior covariance as A * B" * A', but that would be
    # inefficient as we only care about the diagonal. instead, compute this as:
    #
    # diag(A * B" * A') = sum(A * B" . A', -1)
    #
    # which is much faster.

    explained_variance = (np.matmul(C_qt, utils.mprod(BWI_1, BWI_2)) * C_qt).sum(axis=-1)
    posterior_variance = prior_variance - explained_variance

    n_bad, n_tot = (posterior_variance < 0).sum(), posterior_variance.size
    if not n_bad == 0:  # the posterior variance should always be positive
        warnings.warn(f"{n_bad}/{n_tot} information estimates are non-positive.")
        if n_bad / n_tot > 0.5:
            raise ValueError(
                f"More than half ({int(1e2*n_bad/n_tot)}%) of the variance estimates are non-positive."
            )

    marginal_entropy_rate = 0.5 * np.log(2 * np.pi * np.e * posterior_variance)

    return marginal_entropy_rate.sum(axis=-1)


def load(filepath, **kwargs):
    with h5py.File(filepath, "r") as f:
        params = f["params"][:]
    data = pd.read_hdf(filepath, key="data")
    return BayesianOptimizer(init_params=params, init_data=data, **kwargs)



# class BayesianOptimizer:
#     def __init__(
#         self,
#         detectors,
#         dofs,
#         dof_bounds,
#         run_engine,
#         db,
#         shutter=None,
#         fitness_model="tes",
#         init_params=None,
#         init_data=None,
#         init_scheme=None,
#         n_init=None,
#         training_iter=256,
#         verbose=True,
#         **kwargs,
#     ):
#         """
#         A Bayesian optimizer object.

#         detector (Detector)
#         detector_type (str)
#         dofs (list of Devices)
#         dof_bounds (list of bounds)
#         run_engine ()
#         fitness_model (str)

#         training_iter (int)


#         """
#         self.fitness_func = getattr(fitness, fitness_model)
#         self.fitness_args = configs.FITNESS_CONFIGS[fitness_model]['fitness_args']

#         self.dofs, self.dof_bounds = dofs, dof_bounds
#         self.n_dof = len(dofs)

#         self.run_engine = run_engine
#         self.detectors = detectors

#         self.db = db
#         self.training_iter = training_iter

#         self.gridded_plots = True if self.n_dof == 2 else False

#         self.gp_lp_fig = None
#         self.fit_lp_fig = None

#         self.dof_names = np.array([dof.name for dof in self.dofs])

#         MAX_TEST_POINTS = 2**10

#         n_bins_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
#         self.dim_bins = [np.linspace(*bounds, n_bins_per_dim + 1) for bounds in self.dof_bounds]
#         self.dim_mids = [0.5 * (bins[1:] + bins[:-1]) for bins in self.dim_bins]
#         self.test_grid = np.swapaxes(np.r_[np.meshgrid(*self.dim_mids, indexing="ij")], 0, -1)

#         sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
#         self.test_params = sampler.random(n=MAX_TEST_POINTS) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(
#             axis=1
#         )

#         # convert params to x
#         self.params_trans_fun = (
#             lambda params: 2 * (params - self.dof_bounds.min(axis=1)) / self.dof_bounds.ptp(axis=1) - 1
#         )

#         # convert x to params
#         self.inv_params_trans_fun = lambda x: 0.5 * (x + 1) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(
#             axis=1
#         )

#         self.shutter = shutter
#         self.fig, self.axes = None, None

#         if self.shutter is not None:

#             (uid,) = self.run_engine(plans.take_background(self))
#             self.fitness_args['background'] = np.array(list(db[uid].data(field=self.fitness_args['image'])))[0]

#             if self.shutter.status.get() != 0:
#                 raise RuntimeError("Could not open shutter!")

#         else:
#             self.background = 0

#         # for actual prediction and optimization
#         self.evaluator = models.GPR()  # at most 1% of the RMS is due to noise
#         self.validator = models.GPC()
#         self.adjfitGP = models.GPR()

#         self.params = np.zeros((0, self.n_dof))
#         self.data = pd.DataFrame()

#         if (init_params is not None) and (init_data is not None):
#             self.tell(new_params=init_params, new_data=init_data, reuse_hypers=True, verbose=verbose)

#         elif init_scheme == "quasi-random":
#             self.learn(n_iter=1, n_per_iter=n_init, strategy="quasi-random", greedy=True, reuse_hypers=False)

#         else:
#             raise Exception(
#                 "Could not initialize model! Either pass initial params and data, or specify one of:"
#                 "['quasi-random']."
#             )

#     @property
#     def current_params(self):
#         return np.array([dof.read()[dof.name]["value"] for dof in self.dofs])

#     @property
#     def current_X(self):
#         return self.inv_params_trans_fun(self.current_params)

#     @property
#     def optimum(self):
#         return self.inv_params_trans_fun(self.evaluator.X[np.nanargmax(self.evaluator.y)])

#     def go_to_optimum(self):
#         self.run_engine(bps.mv(*[_ for items in zip(self.dofs, np.atleast_1d(self.optimum).T) for _ in items]))

#     def inspect_beam(self, index, masked=True):
#         im = self.masked_images[index] if masked else self.images[index]
#         plt.figure()
#         plt.imshow(im)

#         bbx = self.parsed_image_data.loc[index, ["x_min", "x_max"]].values[[0, 0, 1, 1, 0]]
#         bby = self.parsed_image_data.loc[index, ["y_min", "y_max"]].values[[0, 1, 1, 0, 0]]

#         plt.plot(bbx, bby, lw=1e0, c="r")

#     def save(self, filepath="./model.gpo"):
#         with h5py.File(filepath, "w") as f:
#             f.create_dataset("params", data=self.params)
#         self.data.to_hdf(filepath, key="data")


#     def tell(self, new_params, new_data, reuse_hypers=True, verbose=False):
#         self.params = np.r_[self.params, new_params]
#         self.data = pd.concat([self.data, new_data])

#         X = self.params_trans_fun(self.params)
#         y = self.data.fitness.values
#         c = (~np.isnan(y)).astype(int)

#         self.evaluator.set_data(X[c == 1], y[c == 1])
#         self.validator.set_data(X, c)

#         self.adjfitGP.set_data(X, self.evaluator.mean(X) + np.log(np.maximum(1e-2, self.validator.classify(X))))
#         #self.timer.set_data(np.abs(np.diff(X, axis=0)), self.data.acq_duration.values[1:])

#         #self.timer.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
#         self.evaluator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
#         self.validator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
#         self.adjfitGP.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)

#     def acquire_with_bluesky(self, params, routing=True, verbose=False):
#         if routing:
#             routing_index, _ = utils.get_routing(self.current_params, params)
#             ordered_params = params[routing_index]

#         else:
#             ordered_params = params

#         table = pd.DataFrame(columns=[])

#         #for _params in ordered_params:
#         if verbose:
#             print(f"sampling {ordered_params}")

#         start_params = self.current_params
#         start_time = ttime.monotonic()

#         #try:
#         (uid,) = self.run_engine(
#             bp.list_scan(
#                 self.detectors, *[_ for items in zip(self.dofs, np.atleast_2d(ordered_params).T) for _ in items]
#             )
#         )
#         _table = self.db[uid].table(fill=True)

#         for i, entry in _table.iterrows():

#             keys, vals = self.fitness_func(entry, self.fitness_args)

#             _table.loc[i, keys] = vals
#         # _table.insert(0, "acq_time", ttime.time())
#         # _table.insert(1, "acq_duration", ttime.monotonic() - start_time)
#         # _table.insert(2, "acq_log", "ok")
#         # _table.insert(3, "uid", uid)

#         # except Exception as err:
#         #     warnings.warn(err.args[0])
#         #     columns = ["acq_time", "acq_duration", "acq_log", "uid", f"{self.detector.name}_image"]
#         #     _table = pd.DataFrame(
#         #         [
#         #             (
#         #                 ttime.time(),
#         #                 ttime.monotonic() - start_time,
#         #                 err.args[0],
#         #                 "",
#         #                 np.zeros(self.detector.shape.get()),
#         #             )
#         #         ],
#         #         columns=columns,
#         #     )

#         #for start_param, dof in zip(start_params, self.dofs):
#         #    _table.loc[:, f"delta_{dof.name}"] = dof.read()[dof.name]["value"] - start_param

#         table = pd.concat([table, _table])

#         return ordered_params, table

#     def recommend(
#         self,
#         gp=None,
#         strategy=None,
#         greedy=True,
#         cost_model=None,
#         n=1,
#         n_test=1024,
#         disappointment=1,
#     ):
#         """
#         Recommends the next $n$ points to sample, according to the given strategy.
#         """

#         sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)

#         # this one is easy
#         if strategy.lower() == "quasi-random":
#             return sampler.random(n=n) * 2 - 1

#         if (not greedy) or (n == 1):
#             # recommend some parameters that we might want to sample, with shape (., n, n_dof)
#             TEST_X = (sampler.random(n=n * n_test) * 2 - 1).reshape(n_test, n, self.n_dof)

#             # how much will we have to change our parameters to sample these guys?
#             DELTA_TEST_X = np.diff(
#                 np.concatenate([np.repeat(self.current_X[None, None], n_test, axis=0), TEST_X], axis=1),
#                 axis=1,
#             )

#             if cost_model is None:
#                 cost = np.ones(n_test)

#             # how long will that take?
#             if cost_model == "delay":
#                 cost = self.delay_estimate(DELTA_TEST_X).sum(axis=1)
#                 if not all(cost > 0):
#                     raise ValueError("Some estimated acquisition times are non-positive.")

#             if strategy.lower() == "ei":  # greedy expected reward maximization
#                 objective = -self._negative_expected_improvement(gp, TEST_X).sum(axis=1)

#             if strategy.lower() == "ev":  # greedy expected reward maximization
#                 objective = -self._negative_expected_variance(gp, TEST_X)

#             if strategy.lower() == "ip":
#                 objective = -self._negative_improvement_probability(gp, TEST_X)

#             if strategy.lower() == "a-optimal":
#                 objective = -self._negative_A_optimality(TEST_X)

#             if strategy.lower() == "d-optimal":
#                 objective = -self._negative_D_optimality(TEST_X)

#             return TEST_X[np.argmax(objective / cost)]

#         if greedy and (n > 1):
#             dummy_evaluator = self.evaluator.copy()
#             X_to_sample = np.zeros((0, self.n_dof))

#             for i in range(n):
#                 _X = self.recommend(
#                     strategy=strategy, gp=gp, greedy=True, n=1
#                 )
#                 _y = self.evaluator.mean(_X) - disappointment * self.evaluator.sigma(_X)

#                 X_to_sample = np.r_[X_to_sample, _X]

#                 if not (i + 1 == n):
#                     dummy_evaluator.tell(_X, _y)

#             return X_to_sample

#     def ask(self, **kwargs):
#         return self.inv_params_trans_fun(self.recommend(**kwargs))

#     def learn(
#         self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
#     ):
#         # ip.display.clear_output(wait=True)
#         print(f'learning with strategy "{strategy}" ...')

#         for i in range(n_iter):
#             params_to_sample = np.atleast_2d(
#                 self.ask(n=n_per_iter, strategy=strategy, **kwargs)
#             )  # get point(s) to sample from the strategizer

#             n_original = len(params_to_sample)
#             n_upsample = upsample * n_original + 1

#             upsampled_params_to_sample = sp.interpolate.interp1d(
#                 np.arange(n_original + 1), np.r_[self.current_params[None], params_to_sample], axis=0
#             )(np.linspace(0, n_original, n_upsample)[1:])

#             sampled_params, res_table = self.acquire_with_bluesky(
#                 upsampled_params_to_sample
#             )  # sample the point(s)
#             self.tell(new_params=sampled_params, new_data=res_table, reuse_hypers=reuse_hypers)

#             if "state" in plots:
#                 self.plot_state(remake=False, gridded=self.gridded_plots)
#             if "fitness" in plots:
#                 self.plot_fitness(remake=False)

#             if verbose:
#                 n_params = len(sampled_params)
#                 df_to_print = pd.DataFrame(
#                     np.c_[self.params, self.data.fitness.values], columns=[*self.dof_names, "fitness"]
#                 ).iloc[-n_params:]
#                 print(df_to_print)

#     def plot_fitness(self, remake=True, **kwargs):
#         if (not hasattr(self, "fitness_fig")) or remake:
#             self.make_fitness_plots()

#         self.draw_fitness_plots(**kwargs)

#     def make_fitness_plots(self):
#         """
#         Create the axes onto which we plot/update the cumulative fitness
#         """

#         self.fitness_fig, self.fitness_axes = plt.subplots(
#             1, 1, figsize=(3, 3), dpi=160, sharex=True, sharey=True, constrained_layout=True
#         )
#         self.fitness_axes = np.atleast_2d(self.fitness_axes)

#     def draw_fitness_plots(self):
#         self.fitness_axes[0, 0].clear()

#         # times = self.data.acq_time.astype(int).values / 1e9
#         # times -= times[0]

#         cum_max_fitness = [
#             np.nanmax(self.data.fitness.values[: i + 1]) if not all(np.isnan(self.data.fitness.values[: i + 1])) else np.nan
#             for i in range(len(self.data.fitness.values))
#         ]

#         times = np.arange(len(self.data.fitness.values))

#         ax = self.fitness_axes[0, 0]
#         ax.scatter(times, self.data.fitness.values, c="k", label="fitness samples")
#         ax.plot(times, cum_max_fitness, c="r", label="cumulative best solution")
#         ax.set_xlabel("time (s)")
#         ax.set_ylabel("fitness")
#         ax.legend()

#         self.fitness_fig.canvas.draw_idle()
#         self.fitness_fig.show()

#     def plot_state(self, remake=True, **kwargs):
#         if (not hasattr(self, "state_fig")) or remake:
#             self.make_state_plots()

#         self.draw_state_plots(**kwargs)

#     def make_state_plots(self):
#         """
#         Create the axes onto which we plot/update the state of the GPO
#         """

#         self.state_fig, self.state_axes = plt.subplots(
#             1, 3, figsize=(12, 6), dpi=160, sharex=True, sharey=True, constrained_layout=True
#         )

#         self.state_axes = np.atleast_2d(self.state_axes)

#     def draw_state_plots(self, gridded=False, save_as=None):
#         """
#         Create the axes onto which we plot/update the state of the GPO
#         """

#         s = 16

#         gp = self.adjfitGP

#         test_X = self.params_trans_fun(self.test_params)
#         grid_X = self.params_trans_fun(self.test_grid)

#         # so that the points and estimates have the same nice norm
#         gp_norm = mpl.colors.Normalize(*np.nanpercentile(gp.y, q=[1, 99]))

#         # scatter plot of the fitness for points sampled so far
#         ax = self.state_axes[0, 0]
#         ax.clear()
#         ax.set_title("sampled GP")

#         ref = ax.scatter(*self.inv_params_trans_fun(gp.X).T[:2], s=s, c=gp.y, norm=gp_norm)

#         clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
#         clb.set_label("units")

#         # plot the estimate of test points
#         ax = self.state_axes[0, 1]
#         ax.clear()
#         ax.set_title("GP estimate")
#         if gridded:
#             ref = ax.pcolormesh(
#                 *self.dim_mids[:2], gp.mean(grid_X), norm=gp_norm, shading="nearest"
#             )
#         else:
#             ref = ax.scatter(
#                 *self.test_params.T[:2], s=s, c=gp.mean(test_X), norm=gp_norm
#             )

#         clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
#         clb.set_label("units")
#         #ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

#         #------------------------------------------------------------------------------------------------
#         # plot the entropy rate of test points
#         ax = self.state_axes[0, 2]
#         ax.clear()
#         ax.set_title("GP entropy rate")
#         if gridded:
#             ref = ax.pcolormesh(*self.dim_mids[:2], np.log(gp.sigma(grid_X)), shading="nearest")
#         else:
#             ref = ax.scatter(
#                 *self.test_params.T[:2],
#                 s=s,
#                 c=np.log(gp.sigma(test_X)),
#                 norm=mpl.colors.Normalize(),
#             )

#         clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
#         clb.set_label("nepits per volume")
#         ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

#         for ax in self.state_axes.ravel():
#             ax.set_xlim(*self.dof_bounds[0])
#             ax.set_ylim(*self.dof_bounds[1])

#         self.state_fig.canvas.draw_idle()
#         self.state_fig.show()

#         if save_as is not None:
#             plt.savefig(save_as)

#     # talk to the model

#     def fitness_estimate(self, params):
#         return self.evaluator.mean(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
#             params.shape[:-1]
#         )

#     def fitness_sigma(self, params):
#         return self.evaluator.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
#             params.shape[:-1]
#         )

#     def fitness_entropy(self, params):
#         return np.log(np.sqrt(2 * np.pi * np.e) * self.fitness_sigma(params) + 1e-12)

#     def validate(self, params):
#         return self.validator.classify(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
#             params.shape[:-1]
#         )

#     def delay_estimate(self, params):
#         return self.timer.mean(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

#     def delay_sigma(self, params):
#         return self.timer.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

#     def _negative_improvement_probability(self, gp, test_X):
#         """
#         Returns the negative expected improvement over the maximum, in GP units.
#         """
#         x = test_X.reshape(-1, self.n_dof)

#         # using GPRC units here
#         mu = gp.mean(x)
#         sigma = gp.sigma(x)
#         nu = gp.nu

#         P = 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))

#         return -P.reshape(test_X.shape[:-1])

#     def _negative_expected_improvement(self, gp, test_X):
#         """
#         Returns the negative expected improvement over the maximum, in GP units.
#         """
#         x = test_X.reshape(-1, self.n_dof)

#         # using GPRC units here
#         mu = gp.mean(x)
#         sigma = gp.sigma(x)
#         nu = gp.nu

#         A = np.exp(-0.5 * np.square((mu - nu) / sigma)) / (np.sqrt(2 * np.pi) * sigma)
#         B = 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))
#         E = A * sigma**2 + B * (mu - nu)

#         return E.reshape(test_X.shape[:-1])

#     def _negative_variance(self, gp, test_X):
#         """
#         Returns the negative expected improvement over the maximum, in GP units.
#         """
#         x = test_X.reshape(-1, self.n_dof)

#         sigma = gp.sigma(x)

#         return - (sigma ** 2).reshape(test_X.shape[:-1])

#     def _negative_A_optimality(self, params):
#         """
#         The negative trace of the inverse Fisher information matrix contingent on sampling the passed params.
#         """
#         test_X = self.params_trans_fun(params)
#         invFIM = np.linalg.inv(self.evaluator._contingent_fisher_information_matrix(test_X, delta=1e-3))
#         return np.array(list(map(np.trace, invFIM)))

#     def _negative_D_optimality(self, params):
#         """
#         The negative determinant of the inverse Fisher information matrix contingent on sampling the passed params.
#         """
#         test_X = self.params_trans_fun(params)
#         FIM_stack = self.evaluator._contingent_fisher_information_matrix(test_X, delta=1e-3)
#         return -np.array(list(map(np.linalg.det, FIM_stack)))

import bluesky_adaptive
from bluesky_adaptive.agents.base import Agent

class BayesianOptimizationAgent(Agent):
    def __init__(
        self,
        detectors,
        dofs,
        dof_bounds,
        db,
        shutter=None,
        mode="tes",
        training_iter=256,
        verbose=True,
        sample_center_on_init=True,
    ):
        """
        A Bayesian optimizer object.

        detector (Detector)
        detector_type (str)
        dofs (list of Devices)
        dof_bounds (list of bounds)
        fitness_model (str)

        training_iter (int)


        """

        self.sample_center_on_init = sample_center_on_init

        self.mode = mode
        self.shutter = shutter

        self.config = configs.CONFIGS[mode]

        self.fitness_func = getattr(fitness, mode)
        self.fitness_args = self.config['fitness_args']

        self.dofs, self.dof_bounds = dofs, dof_bounds
        self.n_dof = len(dofs)

        self.detectors = detectors

        self.db = db
        self.training_iter = training_iter

        self.gridded_plots = True if self.n_dof == 2 else False

        self.gp_lp_fig = None
        self.fit_lp_fig = None

        MAX_TEST_POINTS = 2**10

        n_bins_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
        self.dim_bins = [np.linspace(*bounds, n_bins_per_dim + 1) for bounds in self.dof_bounds]
        self.dim_mids = [0.5 * (bins[1:] + bins[:-1]) for bins in self.dim_bins]
        self.test_grid = np.swapaxes(np.r_[np.meshgrid(*self.dim_mids, indexing="ij")], 0, -1)

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.test_params = sampler.random(n=MAX_TEST_POINTS) * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(
            axis=1
        )

        # convert params to x
        self.params_trans_fun = (
            lambda params: (params - self.dof_bounds.min(axis=1)) / self.dof_bounds.ptp(axis=1)
        )

        # convert x to params
        self.inv_params_trans_fun = lambda x: x * self.dof_bounds.ptp(axis=1) + self.dof_bounds.min(axis=1)

        # for actual prediction and optimization
        self.evaluator = models.GPR()  # at most 1% of the RMS is due to noise
        self.validator = models.GPC()
        self.adjfitGP = models.GPR()

        self.params = np.zeros((0, self.n_dof))
        self.data = pd.DataFrame()

    def measurement_plan(self):
        yield from bp.count(detectors=self.detectors)

    def unpack_run(self):
        return None

    def initialize(self,
        init_params=None,
        init_data=None,
        init_scheme=None,
        n_init=4,
    ):
        if self.mode == "tes":
            uid = (yield from plans.take_background(self.shutter, self.detectors))
            self.fitness_args['background'] = np.array(list(self.db[uid].data(field=self.fitness_args['image'])))[0]

            if self.shutter.status.get(as_string=True) != "Open":
                raise RuntimeError(f"Could not open shutter {self.shutter.name}!")
        else:
            self.background = 0

        if (init_params is not None) and (init_data is not None):
            self.tell(new_params=init_params, new_data=init_data, reuse_hypers=True, verbose=self.verbose)

        elif init_scheme == "quasi-random":
            yield from self.learn(n_iter=1, n_per_iter=n_init, strategy=init_scheme, greedy=True, reuse_hypers=False, init=True)

        else:
            raise Exception(
                "Could not initialize model! Either pass initial params and data, or specify one of:"
                "['quasi-random']."
            )

    @property
    def current_params(self):
        return np.array([dof.read()[dof.name]["value"] for dof in self.dofs])

    @property
    def dof_names(self):
        return [dof.name for dof in self.dofs]

    @property
    def det_names(self):
        return self.config['fields']

    @property
    def current_X(self):
        return self.inv_params_trans_fun(self.current_params)

    @property
    def optimum(self):
        return self.inv_params_trans_fun(self.evaluator.X[np.nanargmax(self.evaluator.y)])

    def go_to_optimum(self):
        yield from bps.mv(*[_ for items in zip(self.dofs, np.atleast_1d(self.optimum).T) for _ in items])

    def go_to(self, x):
        yield from bps.mv(*[_ for items in zip(self.dofs, np.atleast_1d(x).T) for _ in items])

    def inspect_beam(self, index, border=None):

        im = self.images[index]

        x_min, x_max, y_min, y_max, width_x, width_y = self.data.loc[index, ['x_min', 'x_max', 'y_min', 'y_max', 'width_x', 'width_y']]

        bbx = np.array([x_min, x_max])[[0, 0, 1, 1, 0]]
        bby = np.array([y_min, y_max])[[0, 1, 1, 0, 0]]

        plt.figure()
        plt.imshow(im, cmap='gray')
        plt.plot(bbx, bby, lw=4e0, c="r")

        if border is not None:
            plt.xlim(x_min - border * width_x, x_min + border * width_x)
            plt.ylim(y_min - border * width_y, y_min + border * width_y)


    def save(self, filepath="./model.gpo"):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("params", data=self.params)
        self.data.to_hdf(filepath, key="data")


    def tell(self, new_params, new_data, reuse_hypers=True, verbose=False):

        self.params = np.r_[self.params, new_params]
        self.data = pd.concat([self.data, new_data])
        self.data.index = np.arange(len(self.data))

        self.images = np.array([im for im in self.data[self.fitness_args['image']].values]) - self.fitness_args['background']

        X = self.params_trans_fun(self.params)
        y = self.data.fitness.values
        c = (~np.isnan(y)).astype(int)

        self.evaluator.set_data(X[c == 1], y[c == 1])
        self.validator.set_data(X, c)

        self.adjfitGP.set_data(X, self.evaluator.mean(X) + np.log(self.validator.classify(X)))
        #self.timer.set_data(np.abs(np.diff(X, axis=0)), self.data.acq_duration.values[1:])

        #self.timer.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.evaluator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.validator.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.adjfitGP.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)

    def acquire_with_bluesky(self, params, routing=True, verbose=False):
        if routing:
            routing_index, _ = utils.get_routing(self.current_params, params)
            ordered_params = params[routing_index]

        else:
            ordered_params = params

        table = pd.DataFrame(columns=[])

        #for _params in ordered_params:
        if verbose:
            print(f"sampling {ordered_params}")

        start_params = self.current_params
        start_time = ttime.monotonic()

        #try:
        uid = yield from bp.list_scan(self.detectors, *[_ for items in zip(self.dofs, np.atleast_2d(ordered_params).T) for _ in items])
        _table = self.db[uid].table(fill=True)
        _table.loc[:, 'uid'] = uid

        for i, entry in _table.iterrows():

            keys, vals = self.fitness_func(entry, self.fitness_args)
            _table.loc[i, keys] = vals

        # _table.insert(0, "acq_time", ttime.time())
        # _table.insert(1, "acq_duration", ttime.monotonic() - start_time)
        # _table.insert(2, "acq_log", "ok")
        # _table.insert(3, "uid", uid)

        # except Exception as err:
        #     warnings.warn(err.args[0])
        #     columns = ["acq_time", "acq_duration", "acq_log", "uid", f"{self.detector.name}_image"]
        #     _table = pd.DataFrame(
        #         [
        #             (
        #                 ttime.time(),
        #                 ttime.monotonic() - start_time,
        #                 err.args[0],
        #                 "",
        #                 np.zeros(self.detector.shape.get()),
        #             )
        #         ],
        #         columns=columns,
        #     )

        #for start_param, dof in zip(start_params, self.dofs):
        #    _table.loc[:, f"delta_{dof.name}"] = dof.read()[dof.name]["value"] - start_param

        table = pd.concat([table, _table])

        return ordered_params, table

    def recommend(
        self,
        evaluator=None,
        validator=None,
        strategy=None,
        greedy=True,
        cost_model=None,
        n=1,
        n_test=1024,
        disappointment=1,
        init=False,
    ):
        """
        Recommends the next $n$ points to sample, according to the given strategy.
        """

        if evaluator is None:
            evaluator = self.evaluator

        if validator is None:
            validator = self.validator

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)

        # this one is easy
        if strategy.lower() == "quasi-random":
            if init and self.sample_center_on_init:
                return np.r_[0.5 * np.ones((1, self.n_dof)), sampler.random(n=n-1)]
            return sampler.random(n=n)

        if (not greedy) or (n == 1):
            # recommend some parameters that we might want to sample, with shape (., n, n_dof)
            TEST_X = sampler.random(n=n * n_test).reshape(n_test, n, self.n_dof)

            # how much will we have to change our parameters to sample these guys?
            DELTA_TEST_X = np.diff(
                np.concatenate([np.repeat(self.current_X[None, None], n_test, axis=0), TEST_X], axis=1),
                axis=1,
            )

            if cost_model is None:
                cost = np.ones(n_test)

            # how long will that take?
            if cost_model == "delay":
                cost = self.delay_estimate(DELTA_TEST_X).sum(axis=1)
                if not all(cost > 0):
                    raise ValueError("Some estimated acquisition times are non-positive.")

            if strategy.lower() == "ei":  # greedy expected reward maximization
                objective = -self._negative_expected_improvement(evaluator, validator, TEST_X).sum(axis=1)

            if strategy.lower() == "ev":  # greedy expected reward maximization
                objective = -self._negative_expected_variance(evaluator, validator, TEST_X).sum(axis=1)

            if strategy.lower() == "eig":
                objective = -_negative_expected_information_gain(evaluator, validator, evaluator.X, TEST_X)

            if strategy.lower() == "ip":
                objective = -self._negative_improvement_probability(evaluator, validator, evaluator.X, TEST_X)

            if strategy.lower() == "a-optimal":
                objective = -self._negative_A_optimality(TEST_X)

            if strategy.lower() == "d-optimal":
                objective = -self._negative_D_optimality(TEST_X)

            return TEST_X[np.argmax(objective / cost)]

        if greedy and (n > 1):
            dummy_evaluator = self.evaluator.copy()
            X_to_sample = np.zeros((0, self.n_dof))

            for i in range(n):
                _X = self.recommend(
                    strategy=strategy, evaluator=dummy_evaluator, validator=validator, greedy=True, n=1
                )
                _y = self.evaluator.mean(_X) - disappointment * self.evaluator.sigma(_X)

                X_to_sample = np.r_[X_to_sample, _X]

                if not (i + 1 == n):
                    dummy_evaluator.tell(_X, _y)

            return X_to_sample

    def ask(self, **kwargs):
        return self.inv_params_trans_fun(self.recommend(**kwargs))

    def learn(
        self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
    ):
        # ip.display.clear_output(wait=True)
        print(f'learning with strategy "{strategy}" ...')

        for i in range(n_iter):
            params_to_sample = np.atleast_2d(
                self.ask(n=n_per_iter, strategy=strategy, **kwargs)
            )  # get point(s) to sample from the strategizer

            n_original = len(params_to_sample)
            n_upsample = upsample * n_original + 1

            upsampled_params_to_sample = sp.interpolate.interp1d(
                np.arange(n_original + 1), np.r_[self.current_params[None], params_to_sample], axis=0
            )(np.linspace(0, n_original, n_upsample)[1:])

            sampled_params, res_table = yield from self.acquire_with_bluesky(
                upsampled_params_to_sample
            )  # sample the point(s)

            self.tell(new_params=sampled_params, new_data=res_table, reuse_hypers=reuse_hypers)

            if "state" in plots:
                self.plot_state(remake=False, gridded=self.gridded_plots)
            if "fitness" in plots:
                self.plot_fitness(remake=False)

            if verbose:
                n_params = len(sampled_params)
                df_to_print = pd.DataFrame(
                    np.c_[self.params, self.data.fitness.values], columns=[*self.dof_names, "fitness"]
                ).iloc[-n_params:]
                print(df_to_print)

    def plot_fitness(self, remake=True, **kwargs):
        if (not hasattr(self, "fitness_fig")) or remake:
            self.make_fitness_plots()

        self.draw_fitness_plots(**kwargs)

    def make_fitness_plots(self):
        """
        Create the axes onto which we plot/update the cumulative fitness
        """

        self.fitness_fig, self.fitness_axes = plt.subplots(
            1, 1, figsize=(3, 3), dpi=160, sharex=True, sharey=True, constrained_layout=True
        )
        self.fitness_axes = np.atleast_2d(self.fitness_axes)

    def draw_fitness_plots(self):
        self.fitness_axes[0, 0].clear()

        # times = self.data.acq_time.astype(int).values / 1e9
        # times -= times[0]

        cum_max_fitness = [
            np.nanmax(self.data.fitness.values[: i + 1]) if not all(np.isnan(self.data.fitness.values[: i + 1])) else np.nan
            for i in range(len(self.data.fitness.values))
        ]

        times = np.arange(len(self.data.fitness.values))

        ax = self.fitness_axes[0, 0]
        ax.scatter(times, self.data.fitness.values, c="k", label="fitness samples")
        ax.plot(times, cum_max_fitness, c="r", label="cumulative best solution")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("fitness")
        ax.legend()

        self.fitness_fig.canvas.draw_idle()
        self.fitness_fig.show()

    def plot_state(self, remake=True, **kwargs):
        if (not hasattr(self, "state_fig")) or remake:
            self.make_state_plots()

        self.draw_state_plots(**kwargs)

    def make_state_plots(self):
        """
        Create the axes onto which we plot/update the state of the GPO
        """

        self.state_fig, self.state_axes = plt.subplots(
            2, 5, figsize=(12, 6), dpi=160, sharex=True, sharey=True, constrained_layout=True
        )

    def draw_state_plots(self, gridded=False, save_as=None):
        """
        Create the axes onto which we plot/update the state of the GPO
        """

        s = 16

        if gridded:
            P = self.validate(self.test_grid)
        else:
            P = self.validate(self.test_params)

        PE = 0.5 * np.log(2 * np.pi * np.e * P * (1 - P))

        # so that the points and estimates have the same nice norm
        fitness_norm = mpl.colors.Normalize(*np.nanpercentile(self.data.fitness.values, q=[1, 99]))

        # scatter plot of the fitness for points sampled so far
        ax = self.state_axes[0, 0]
        ax.clear()
        ax.set_title("sampled fitness")

        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ref = ax.scatter(*self.params.T[:2], s=s, c=self.data.fitness.values, norm=fitness_norm)

        max_improvement_params = self.inv_params_trans_fun(
            self.recommend(strategy="ei", greedy=True, n=8, n_test=1024)
        )
        max_information_params = self.inv_params_trans_fun(
            self.recommend(strategy="ev", greedy=True, n=8, n_test=1024)
        )

        max_improvement_routing_index, _ = utils.get_routing(self.current_params, max_improvement_params)
        ordered_max_improvement_params = max_improvement_params[max_improvement_routing_index]

        max_information_routing_index, _ = utils.get_routing(self.current_params, max_information_params)
        ordered_max_information_params = max_information_params[max_information_routing_index]

        # ax.legend(fontsize=6)

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("fitness units")

        # plot the estimate of test points
        ax = self.state_axes[0, 1]
        ax.clear()
        ax.set_title("fitness estimate")
        if gridded:
            ref = ax.pcolormesh(
                *self.dim_mids[:2], self.fitness_estimate(self.test_grid), norm=fitness_norm, shading="nearest"
            )
        else:
            ref = ax.scatter(
                *self.test_params.T[:2], s=s, c=self.fitness_estimate(self.test_params), norm=fitness_norm
            )

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("fitness units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot the entropy rate of test points
        ax = self.state_axes[0, 2]
        ax.clear()
        ax.set_title("fitness entropy rate")
        if gridded:
            ref = ax.pcolormesh(*self.dim_mids[:2], self.fitness_entropy(self.test_grid), shading="nearest")
        else:
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=self.fitness_entropy(self.test_params),
                norm=mpl.colors.Normalize(),
            )

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("nepits per volume")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot the estimate of test points
        ax = self.state_axes[0, 3]
        ax.clear()
        ax.set_title("expected improvement")

        if gridded:
            expected_improvement = -self._negative_expected_improvement(
                self.evaluator, self.validator, self.params_trans_fun(self.test_grid)
            )
            expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.pcolormesh(
                *self.dim_mids[:2], expected_improvement, norm=mpl.colors.Normalize(vmin=0), shading="nearest"
            )
        else:
            expected_improvement = -self._negative_expected_improvement(
                self.evaluator, self.validator, self.params_trans_fun(self.test_params)
            )
            expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=-expected_improvement,
                norm=mpl.colors.Normalize(vmin=0),
            )

        ax.plot(*ordered_max_improvement_params.T[:2], lw=1, color="k")
        ax.scatter(*ordered_max_improvement_params.T[:2], marker="*", color="k", s=s, label="max_improvement")

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("fitness units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot classification of data points
        ax = self.state_axes[1, 0]
        ax.clear()
        ax.set_title("sampled validity")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ref = ax.scatter(*self.params.T[:2], s=s, c=self.validator.c, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_ticks([0, 1])
        clb.set_ticklabels(["invalid", "valid"])

        ax = self.state_axes[1, 1]
        ax.clear()
        ax.set_title("validity estimate")
        if gridded:
            ref = ax.pcolormesh(*self.dim_mids[:2], P, vmin=0, vmax=1, shading="nearest")
        else:
            ref = ax.scatter(*self.test_params.T[:2], s=s, c=P, vmin=0, vmax=1)
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_ticks([0, 1])
        clb.set_ticklabels(["invalid", "valid"])
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ax = self.state_axes[1, 2]
        ax.set_title("validity entropy rate")
        if gridded:
            ref = ax.pcolormesh(*self.dim_mids[:2], PE, shading="nearest")
        else:
            ref = ax.scatter(*self.test_params.T[:2], s=s, c=PE)
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("nepits per volume")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ax = self.state_axes[1, 3]
        ax.clear()
        ax.set_title("information gain")
        if gridded:
            expected_information_gain = -self._negative_expected_variance(
                self.evaluator,
                self.validator,
                self.params_trans_fun(self.test_grid).reshape(-1, 1, self.n_dof),
            ).reshape(self.test_grid.shape[:2])
            expected_information_gain[~(expected_information_gain > 0)] = np.nan
            ref = ax.pcolormesh(
                *self.dim_mids[:2],
                expected_information_gain,
                norm=mpl.colors.Normalize(vmin=0),
                shading="nearest",
            )
        else:
            expected_information_gain = -self._negative_expected_variance(
                self.evaluator,
                self.validator,
                self.params_trans_fun(self.test_params[:, None, :]),
            )
            expected_information_gain[~(expected_information_gain > 0)] = np.nan
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=expected_information_gain,
                norm=mpl.colors.Normalize(vmin=0),
            )

        ax.plot(*ordered_max_information_params.T[:2], lw=1, color="k")
        ax.scatter(*ordered_max_information_params.T[:2], marker="s", color="k", s=s, label="max_information")

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        clb.set_label("total nepits")

        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ax = self.state_axes[1, 4]
        ax.clear()
        ax.set_title("cFIM determinant")
        if gridded:
            D_optimality = -self._negative_D_optimality(self.test_grid.reshape(-1, 1, self.n_dof)).reshape(
                self.test_grid.shape[:2]
            )
            ref = ax.pcolormesh(
                *self.dim_mids[:2],
                D_optimality,
                shading="nearest",
            )
        else:
            D_optimality = -self._negative_D_optimality(self.test_params)
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=D_optimality,
            )

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32)
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        for ax in self.state_axes.ravel():
            ax.set_xlim(*self.dof_bounds[0])
            ax.set_ylim(*self.dof_bounds[1])

        self.state_fig.canvas.draw_idle()
        self.state_fig.show()

        if save_as is not None:
            plt.savefig(save_as)

    # talk to the model

    def fitness_estimate(self, params):
        return self.evaluator.mean(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
            params.shape[:-1]
        )

    def fitness_sigma(self, params):
        return self.evaluator.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(
            params.shape[:-1]
        )

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

    def _negative_improvement_probability(self, evaluator, validator, test_X):
        """
        Returns the negative expected improvement over the maximum, in GP units.
        """
        x = test_X.reshape(-1, self.n_dof)

        # using GPRC units here
        mu = evaluator.mean(x)
        sigma = evaluator.sigma(x)
        nu = evaluator.nu
        pv = validator.classify(x)

        P = pv * 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))

        return -P.reshape(test_X.shape[:-1])

    def _negative_expected_improvement(self, evaluator, validator, test_X):
        """
        Returns the negative expected improvement over the maximum, in GP units.
        """
        x = test_X.reshape(-1, self.n_dof)

        # using GPRC units here
        mu = evaluator.mean(x)
        sigma = evaluator.sigma(x)
        nu = evaluator.nu
        pv = validator.classify(x)

        A = np.exp(-0.5 * np.square((mu - nu) / sigma)) / (np.sqrt(2 * np.pi) * sigma)
        B = 0.5 * (1 + sp.special.erf((mu - nu) / (np.sqrt(2) * sigma)))
        E = -pv * (A * sigma**2 + B * (mu - nu))

        return E.reshape(test_X.shape[:-1])

    def _negative_expected_variance(self, evaluator, validator, test_X):
        """
        Returns the negative expected improvement over the maximum, in GP units.
        """
        x = test_X.reshape(-1, self.n_dof)

        sigma = evaluator.sigma(x)
        pv = validator.classify(x)

        return - (pv * sigma ** 2).reshape(test_X.shape[:-1])

    def _negative_A_optimality(self, params):
        """
        The negative trace of the inverse Fisher information matrix contingent on sampling the passed params.
        """
        test_X = self.params_trans_fun(params)
        invFIM = np.linalg.inv(self.evaluator._contingent_fisher_information_matrix(test_X, delta=1e-3))
        return np.array(list(map(np.trace, invFIM)))

    def _negative_D_optimality(self, params):
        """
        The negative determinant of the inverse Fisher information matrix contingent on sampling the passed params.
        """
        test_X = self.params_trans_fun(params)
        FIM_stack = self.evaluator._contingent_fisher_information_matrix(test_X, delta=1e-3)
        return -np.array(list(map(np.linalg.det, FIM_stack)))
