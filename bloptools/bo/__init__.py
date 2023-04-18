import logging

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

from .. import utils
from . import acquisition, models

mpl.rc("image", cmap="coolwarm")


def load(filepath, **kwargs):
    with h5py.File(filepath, "r") as f:
        params = f["params"][:]
    data = pd.read_hdf(filepath, key="data")
    return BayesianOptimizationAgent(init_params=params, init_data=data, **kwargs)


# import bluesky_adaptive
# from bluesky_adaptive.agents.base import Agent


class BayesianOptimizationAgent:
    def __init__(
        self,
        dofs,
        dets,
        bounds,
        experiment,
        db,
        training_iter=256,
        verbose=True,
        sample_center_on_init=True,
    ):
        """
        A Bayesian optimizer object.

        detector (Detector)
        detector_type (str)
        dofs (list of Devices)
        bounds (list of bounds)
        fitness_model (str)

        training_iter (int)


        """

        self.dofs = dofs
        for dof in self.dofs:
            dof.kind = "hinted"

        self.n_dof = len(dofs)

        self.bounds = bounds if bounds is not None else np.array([[-1.0, +1.0] for i in range(self.n_dof)])

        self.dets = dets

        self.experiment = experiment

        self.db = db
        self.training_iter = training_iter

        self.gridded_plots = True if self.n_dof == 2 else False

        MAX_TEST_POINTS = 2**10

        n_bins_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
        self.dim_bins = [np.linspace(*bounds, n_bins_per_dim + 1) for bounds in self.bounds]
        self.dim_mids = [0.5 * (bins[1:] + bins[:-1]) for bins in self.dim_bins]
        self.test_grid = np.swapaxes(np.r_[np.meshgrid(*self.dim_mids, indexing="ij")], 0, -1)

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.test_params = sampler.random(n=MAX_TEST_POINTS) * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

        # convert params to x
        self.params_trans_fun = lambda params: (params - self.bounds.min(axis=1)) / self.bounds.ptp(axis=1)

        # convert x to params
        self.inv_params_trans_fun = lambda x: x * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

        # for actual prediction and optimization
        self.params = np.empty((0, self.n_dof))
        self.data = pd.DataFrame()

        self.evaluator = models.GPR(MIN_SNR=experiment.MIN_SNR)
        self.classifier = models.GPC()

    def measurement_plan(self):
        yield from bp.count(detectors=self.dets)

    def unpack_run(self):
        return None

    def initialize(
        self,
        init_params=None,
        init_data=None,
        init_scheme=None,
        n_init=4,
    ):
        # experiment-specific stuff
        yield from self.experiment.initialize()

        # now let's get bayesian
        if (init_params is not None) and (init_data is not None):
            self.tell(new_params=init_params, new_data=init_data, reuse_hypers=True, verbose=self.verbose)

        elif init_scheme == "quasi-random":
            yield from self.learn(
                n_iter=1, n_per_iter=n_init, strategy=init_scheme, greedy=True, reuse_hypers=False, init=True
            )

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
        return self.experiment.DEPENDENT_COMPONENTS

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

        x_min, x_max, y_min, y_max, width_x, width_y = self.data.loc[
            index, ["x_min", "x_max", "y_min", "y_max", "width_x", "width_y"]
        ]

        bbx = np.array([x_min, x_max])[[0, 0, 1, 1, 0]]
        bby = np.array([y_min, y_max])[[0, 1, 1, 0, 0]]

        plt.figure()
        plt.imshow(im, cmap="gray")
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

        if hasattr(self.experiment, "IMAGE_NAME"):
            self.images = np.array([im for im in self.data[self.experiment.IMAGE_NAME].values])

        X = self.params_trans_fun(self.params)
        Y = self.data.fitness.values[:, None]
        c = (~np.isnan(Y).any(axis=-1)).astype(int)

        self.evaluator.set_data(X[c == 1], Y[c == 1])
        self.classifier.set_data(X, c)

        # self.timer.train(training_iter=self.training_iter, reuse_hypers=reuse_hypers, verbose=verbose)
        self.evaluator.train(maxiter=self.training_iter)
        self.classifier.train(maxiter=self.training_iter)

    def acquire_with_bluesky(self, params, routing=True, verbose=False):
        if routing:
            routing_index, _ = utils.get_routing(self.current_params, params)
            ordered_params = params[routing_index]

        else:
            ordered_params = params

        table = pd.DataFrame(columns=[])

        # for _params in ordered_params:
        if verbose:
            print(f"sampling {ordered_params}")

        try:
            uid = yield from bp.list_scan(
                self.dets, *[_ for items in zip(self.dofs, np.atleast_2d(ordered_params).T) for _ in items]
            )
            _table = self.db[uid].table(fill=True)
            _table.loc[:, "uid"] = uid
        except Exception as err:
            logging.warning(repr(err))

        for i, entry in _table.iterrows():
            keys, vals = self.experiment.parse_entry(entry)
            _table.loc[i, keys] = vals

        table = pd.concat([table, _table])

        return ordered_params, table

    def recommend(
        self,
        evaluator=None,
        classifier=None,
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

        if classifier is None:
            classifier = self.classifier

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)

        # this one is easy
        if strategy.lower() == "quasi-random":
            # if init and self.sample_center_on_init:
            #    return np.r_[0.5 * np.ones((1, self.n_dof)), sampler.random(n=n-1)]
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

            if strategy.lower() == "ei":  # maximize the expected improvement
                objective = acquisition.expected_improvement(evaluator, classifier, TEST_X).sum(axis=1)

            if strategy.lower() == "max_entropy":  # maximize the total entropy
                objective = (evaluator.normalized_entropy(TEST_X) + classifier.entropy(TEST_X)).sum(axis=1)

            if strategy.lower() == "egibbon":  # maximize the expected GIBBON
                objective = acquisition.expected_gibbon(evaluator, classifier, TEST_X).sum(axis=1)

            return TEST_X[np.argmax(objective / cost)]

        if greedy and (n > 1):
            dummy_evaluator = self.evaluator.copy()
            X_to_sample = np.zeros((0, self.n_dof))

            for i in range(n):
                _X = self.recommend(
                    strategy=strategy, evaluator=dummy_evaluator, classifier=classifier, greedy=True, n=1
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
            np.nanmax(self.data.fitness.values[: i + 1])
            if not all(np.isnan(self.data.fitness.values[: i + 1]))
            else np.nan
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

    def plot_state(self, remake=True, gridded=False, save_as=None):
        if (not hasattr(self, "state_fig")) or remake:
            self.make_state_plots()

        self.update_state_plots(gridded=gridded)
        self.show_state_plots(save_as=save_as)

    def make_state_plots(self):
        """
        Create the axes onto which we plot/update the state of the GPO
        """

        self.state_fig, self.state_axes = plt.subplots(
            2, 4, figsize=(10, 6), dpi=160, sharex=True, sharey=True, constrained_layout=True
        )

    def update_state_plots(self, gridded=False):
        """
        Create the axes onto which we plot/update the state of the GPO
        """

        s = 16

        # so that the points and estimates have the same nice norm
        fitness_norm = mpl.colors.Normalize(*np.nanpercentile(self.data.fitness.values, q=[1, 99]))

        # scatter plot of the fitness for points sampled so far
        ax = self.state_axes[0, 0]
        ax.clear()
        ax.set_title("sampled fitness")

        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ref = ax.scatter(*self.params.T[:2], s=s, c=self.data.fitness.values, norm=fitness_norm)

        # max_improvement_params = self.inv_params_trans_fun(
        #     self.recommend(strategy="exploit", greedy=True, n=8, n_test=1024)
        # )
        # max_information_params = self.inv_params_trans_fun(
        #     self.recommend(strategy="ev", greedy=True, n=8, n_test=1024)
        # )

        # max_improvement_routing_index, _ = utils.get_routing(self.current_params, max_improvement_params)
        # ordered_max_improvement_params = max_improvement_params[max_improvement_routing_index]

        # max_information_routing_index, _ = utils.get_routing(self.current_params, max_information_params)
        # ordered_max_information_params = max_information_params[max_information_routing_index]

        # ax.legend(fontsize=6)

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
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

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("fitness units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot the entropy rate of test points
        ax = self.state_axes[0, 2]
        ax.clear()
        ax.set_title("fitness uncertainty")
        if gridded:
            ref = ax.pcolormesh(*self.dim_mids[:2], self.fitness_sigma(self.test_grid), shading="nearest")
        else:
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=self.fitness_sigma(self.test_params),
                norm=mpl.colors.LogNorm(),
            )

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("fitness units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot the estimate of test points
        ax = self.state_axes[0, 3]
        ax.clear()
        ax.set_title("expected improvement")

        if gridded:
            expected_improvement = acquisition.expected_improvement(
                self.evaluator, self.classifier, self.params_trans_fun(self.test_grid)
            )
            # expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.pcolormesh(
                *self.dim_mids[:2], expected_improvement, norm=mpl.colors.Normalize(), shading="nearest"
            )
        else:
            expected_improvement = acquisition.expected_improvement(
                self.evaluator, self.classifier, self.params_trans_fun(self.test_params)
            )
            # expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=expected_improvement,
                norm=mpl.colors.Normalize(),
            )

        # ax.plot(*ordered_max_improvement_params.T[:2], lw=1, color="k")
        # ax.scatter(*ordered_max_improvement_params.T[:2], marker="*", color="k", s=s, label="max_improvement")

        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("fitness units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        # plot classification of data points
        ax = self.state_axes[1, 0]
        ax.clear()
        ax.set_title("sampled validity")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ref = ax.scatter(*self.params.T[:2], s=s, c=self.classifier.c, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_ticks([0, 1])
        clb.set_ticklabels(["invalid", "valid"])

        ax = self.state_axes[1, 1]
        ax.clear()
        ax.set_title("validity estimate")
        if gridded:
            ref = ax.pcolormesh(
                *self.dim_mids[:2],
                self.classifier.p(self.params_trans_fun(self.test_grid)),
                vmin=0,
                vmax=1,
                shading="nearest",
            )
        else:
            ref = ax.scatter(
                *self.test_params.T[:2],
                c=self.classifier.p(self.params_trans_fun(self.test_params)),
                s=s,
                vmin=0,
                vmax=1,
            )
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("probability of validity")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ax = self.state_axes[1, 2]
        ax.clear()
        ax.set_title("validity entropy")
        if gridded:
            ref = ax.pcolormesh(
                *self.dim_mids[:2],
                self.classifier.entropy(self.params_trans_fun(self.test_grid)),
                shading="nearest",
            )
        else:
            ref = ax.scatter(
                *self.test_params.T[:2], c=self.classifier.entropy(self.params_trans_fun(self.test_params)), s=s
            )
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("nats")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        ax = self.state_axes[1, 3]
        ax.clear()
        ax.set_title("expected GIBBON")
        if gridded:
            expected_improvement = acquisition.expected_gibbon(
                self.evaluator, self.classifier, self.params_trans_fun(self.test_grid)
            )
            # expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.pcolormesh(
                *self.dim_mids[:2], expected_improvement, norm=mpl.colors.Normalize(), shading="nearest"
            )
        else:
            expected_improvement = acquisition.expected_gibbon(
                self.evaluator, self.classifier, self.params_trans_fun(self.test_params)
            )
            # expected_improvement[~(expected_improvement > 0)] = np.nan
            ref = ax.scatter(
                *self.test_params.T[:2],
                s=s,
                c=expected_improvement,
                norm=mpl.colors.Normalize(),
            )
        clb = self.state_fig.colorbar(ref, ax=ax, location="bottom", aspect=32, shrink=0.8)
        clb.set_label("units")
        ax.scatter(*self.params.T[:2], s=s, edgecolor="k", facecolor="none")

        for ax in self.state_axes.ravel():
            ax.set_xlim(*self.bounds[0])
            ax.set_ylim(*self.bounds[1])

    def show_state_plots(self, save_as=None):
        self.state_fig.canvas.draw_idle()
        self.state_fig.show()

        if save_as is not None:
            plt.savefig(save_as, dpi=256)

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
        return self.classifier.p(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

    def delay_estimate(self, params):
        return self.timer.mean(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

    def delay_sigma(self, params):
        return self.timer.sigma(self.params_trans_fun(params).reshape(-1, self.n_dof)).reshape(params.shape[:-1])

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
