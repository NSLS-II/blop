import copy
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
        X = f["X"][:]
    data = pd.read_hdf(filepath, key="data")
    return BayesianOptimizationAgent(init_X=X, init_data=data, **kwargs)


# import bluesky_adaptive
# from bluesky_adaptive.agents.base import Agent


class BayesianOptimizationAgent:
    def __init__(
        self,
        dofs,
        dets,
        bounds,
        experiment,
        tasks,
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

        self.tasks = tasks

        self.num_tasks = len(tasks)

        self._initialized = False

        self.db = db
        self.training_iter = training_iter

        MAX_TEST_POINTS = 2**10

        n_bins_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
        self.dim_bins = np.array([np.linspace(*bounds, n_bins_per_dim + 1) for bounds in self.bounds])
        self.dim_mids = np.array([0.5 * (bins[1:] + bins[:-1]) for bins in self.dim_bins])
        self.test_X_grid = np.swapaxes(np.r_[np.meshgrid(*self.dim_mids, indexing="ij")], 0, -1)

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.test_X = sampler.random(n=MAX_TEST_POINTS) * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

        # convert X to x
        self.X_trans_fun = lambda X: (X - self.bounds.min(axis=1)) / self.bounds.ptp(axis=1)

        # convert x to X
        self.inv_X_trans_fun = lambda x: x * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

        # for actual prediction and optimization
        self.X = np.empty((0, self.n_dof))
        self.data = pd.DataFrame()

        for task in self.tasks:
            task.regressor = models.GPR(bounds=self.bounds, MIN_SNR=task.MIN_SNR)

        # self.regressor = models.GPR(bounds=self.bounds, MIN_SNR=experiment.MIN_SNR)
        self.classifier = models.GPC(bounds=self.bounds)

    def measurement_plan(self):
        yield from bp.count(detectors=self.dets)

    def unpack_run(self):
        return None

    # def load(filepath, **kwargs):
    # with h5py.File(filepath, "r") as f:
    #     X = f["X"][:]
    # data = pd.read_hdf(filepath, key="data")
    # return BayesianOptimizationAgent(init_X=X, init_data=data, **kwargs)

    def initialize(
        self,
        filepath=None,
        init_X=None,
        init_data=None,
        init_scheme=None,
        n_init=4,
    ):
        if filepath is not None:
            with h5py.File(filepath, "r") as f:
                X = f["X"][:]
            data = pd.read_hdf(filepath, key="data")
            self.tell(new_X=X, new_data=data)
            return

        # experiment-specific stuff
        yield from self.experiment.initialize()

        # now let's get bayesian
        if (init_X is not None) and (init_data is not None):
            self.tell(new_X=init_X, new_data=init_data, reuse_hypers=True, verbose=self.verbose)

        elif init_scheme == "quasi-random":
            yield from self.learn(
                n_iter=1, n_per_iter=n_init, strategy=init_scheme, greedy=True, reuse_hypers=False, route=True
            )

        else:
            raise Exception(
                "Could not initialize model! Either pass initial X and data, or specify one of:"
                "['quasi-random']."
            )

        self._initialized = True

    @property
    def current_X(self):
        return np.array([dof.read()[dof.name]["value"] for dof in self.dofs])

    @property
    def dof_names(self):
        return [dof.name for dof in self.dofs]

    @property
    def det_names(self):
        return self.experiment.DEPENDENT_COMPONENTS

    @property
    def optimum(self):
        return self.regressor.X[np.argmax(self.regressor.Y)]

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
        plt.imshow(im, cmap="gray_r")
        plt.plot(bbx, bby, lw=4e0, c="r")

        if border is not None:
            plt.xlim(x_min - border * width_x, x_min + border * width_x)
            plt.ylim(y_min - border * width_y, y_min + border * width_y)

    def save(self, filepath="model.h5"):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("X", data=self.X)
        self.data.to_hdf(filepath, key="data")

    def tell(self, new_X, new_data, reuse_hypers=True, verbose=False):
        self.X = np.r_[self.X, new_X]
        self.data = pd.concat([self.data, new_data])
        self.data.index = np.arange(len(self.data))

        for index, entry in self.data.iterrows():
            total_fitness = 0
            for task in self.tasks:
                task_fitness = task.get_fitness(entry)
                self.data.loc[index, f"{task.name}_fitness"] = task_fitness
                total_fitness += task_fitness
            self.data.loc[index, "tasks_sum"] = total_fitness

        if hasattr(self.experiment, "IMAGE_NAME"):
            self.images = np.array([im for im in self.data[self.experiment.IMAGE_NAME].values])

        X = self.X.copy()
        Y = np.c_[[getattr(self.data, f"{task.name}_fitness").values for task in self.tasks]].T

        for task in self.tasks:
            X = self.X.copy()
            Y = getattr(self.data, f"{task.name}_fitness").values[:, None]
            c = (~np.isnan(Y).any(axis=-1)).astype(int)

            task.regressor.set_data(X[c == 1], Y[c == 1])
            task.regressor.train(step_limit=self.training_iter)

        # self.regressor.set_data(X, Y)
        # self.regressor.train(step_limit=self.training_iter)

        c = (~np.isnan(Y).any(axis=-1)).astype(int)
        self.classifier.set_data(X, c)
        self.classifier.train(step_limit=self.training_iter)

    def acquire_with_bluesky(self, X, verbose=False):
        # for _X in ordered_X:
        if verbose:
            print(f"sampling {X}")

        try:
            uid = yield from bp.list_scan(
                self.dets, *[_ for items in zip(self.dofs, np.atleast_2d(X).T) for _ in items]
            )

            table = self.db[uid].table(fill=True)
            table.loc[:, "uid"] = uid

        except Exception as err:
            table = pd.DataFrame()
            logging.warning(repr(err))

        for index, entry in table.iterrows():
            for k, v in self.experiment.postprocess(entry).items():
                table.loc[index, k] = v

        return X, table

    def qr_sample(self, n):
        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        return sampler.random(n=n) * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

    def sample_acqf(self, acqf, n_test=2048, optimize=False):
        def acq_loss(x, *args):
            return -acqf(x, *args)

        acq_args = (self.tasks, self.classifier)

        test_X = self.qr_sample(n=n_test)
        init_X = test_X[acq_loss(test_X, *acq_args).argmin()]

        # print(init_X)

        if optimize:
            res = sp.optimize.minimize(
                fun=acq_loss, args=acq_args, x0=init_X, bounds=self.bounds, method="SLSQP", options={"maxiter": 64}
            )
            X = res.x
        else:
            X = init_X

        # print(res)

        return X, acq_loss(X, *acq_args)

    def ask(
        self,
        tasks=None,
        classifier=None,
        strategy=None,
        greedy=True,
        n=1,
        disappointment=0,
        route=True,
        cost_model=None,
        n_test=1024,
        optimize=True,
    ):
        """
        Recommends the next $n$ points to sample.
        """

        if route:
            unrouted_X = self.ask(
                tasks=tasks,
                classifier=classifier,
                strategy=strategy,
                greedy=greedy,
                n=n,
                disappointment=disappointment,
                route=False,
                cost_model=cost_model,
                n_test=n_test,
            )

            routing_index, _ = utils.get_routing(self.current_X, unrouted_X)
            return unrouted_X[routing_index]

        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)

        # this one is easy
        if strategy.lower() == "quasi-random":
            # if init and self.sample_center_on_init:
            #    return np.r_[0.5 * np.ones((1, self.dim)), sampler.random(n=n-1)]
            return sampler.random(n=n) * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

        if not self._initialized:
            raise RuntimeError('An uninitialized agent only accepts the strategy "quasi-random".')

        if tasks is None:
            tasks = self.tasks

        if classifier is None:
            classifier = self.classifier

        if (not greedy) or (n == 1):
            acqf = None

            if strategy.lower() == "est":  # maximize the expected improvement
                acqf = acquisition.expected_sum_of_tasks

            if strategy.lower() == "esti":  # maximize the expected improvement
                acqf = acquisition.expected_sum_of_tasks_improvement

            if acqf is None:
                raise ValueError(f'Unrecognized strategy "{strategy}".')

            X, loss = self.sample_acqf(acqf, optimize=optimize)

            return np.atleast_2d(X)

        if greedy and (n > 1):
            dummy_tasks = [copy.deepcopy(task) for task in self.tasks]

            X_to_sample = np.zeros((0, self.n_dof))

            for i in range(n):
                _X = self.ask(strategy=strategy, tasks=dummy_tasks, classifier=classifier, greedy=True, n=1)

                X_to_sample = np.r_[X_to_sample, _X]

                _y = np.c_[
                    [task.regressor.mean(_X) - disappointment * task.regressor.sigma(_X) for task in dummy_tasks]
                ].T

                if not (i + 1 == n):
                    for task in dummy_tasks:
                        _y = task.regressor.mean(_X) - disappointment * task.regressor.sigma(_X)
                        task.regressor.tell(_X, _y)

            return X_to_sample

    def learn(
        self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
    ):
        # ip.display.clear_output(wait=True)
        print(f'learning with strategy "{strategy}" ...')

        for i in range(n_iter):
            X_to_sample = np.atleast_2d(
                self.ask(n=n_per_iter, strategy=strategy, **kwargs)
            )  # get point(s) to sample from the strategizer

            n_original = len(X_to_sample)
            n_upsample = upsample * n_original + 1

            upsampled_X_to_sample = sp.interpolate.interp1d(
                np.arange(n_original + 1), np.r_[self.current_X[None], X_to_sample], axis=0
            )(np.linspace(0, n_original, n_upsample)[1:])

            sampled_X, res_table = yield from self.acquire_with_bluesky(
                upsampled_X_to_sample
            )  # sample the point(s)

            self.tell(new_X=sampled_X, new_data=res_table, reuse_hypers=reuse_hypers)

            if "state" in plots:
                self.plot_state(remake=False, gridded=self.gridded_plots)
            if "fitness" in plots:
                self.plot_fitness(remake=False)

            # if verbose:
            #     n_X = len(sampled_X)
            #     df_to_print = pd.DataFrame(
            #         np.c_[self.X, self.experiment.HINTED_STATS],
            #         columns=[*self.dof_names, *self.experiment.HINTED_STATS]
            #     ).iloc[-n_X:]
            #     print(df_to_print)

    def plot_constraints(self, axes=[0, 1]):
        s = 32

        gridded = self.n_dof == 2

        self.class_fig, self.class_axes = plt.subplots(
            1, 3, figsize=(7, 2), sharex=True, sharey=True, constrained_layout=True
        )

        for ax in self.class_axes.ravel():
            ax.set_xlabel(self.dofs[axes[0]].name)
            ax.set_ylabel(self.dofs[axes[1]].name)

        data_ax = self.class_axes[0].scatter(
            *self.classifier.X.T[:2], s=s, c=self.classifier.c, vmin=0, vmax=1, cmap="plasma"
        )

        if gridded:
            self.class_axes[1].pcolormesh(
                *self.dim_mids[axes],
                self.classifier.p(self.test_X_grid),
                shading="nearest",
                cmap="plasma",
                vmin=0,
                vmax=1,
            )
            entropy_ax = self.class_axes[2].pcolormesh(
                *self.dim_mids[axes], self.classifier.entropy(self.test_X_grid), shading="nearest", cmap="plasma"
            )

        else:
            self.class_axes[1].scatter(
                *self.test_X.T[axes], s=s, c=self.classifier.p(self.test_X), vmin=0, vmax=1, cmap="plasma"
            )
            entropy_ax = self.class_axes[2].scatter(
                *self.test_X.T[axes], s=s, c=self.classifier.entropy(self.test_X), cmap="plasma"
            )

        self.class_fig.colorbar(data_ax, ax=self.class_axes[:2], location="bottom", aspect=32, shrink=0.8)
        self.class_fig.colorbar(entropy_ax, ax=self.class_axes[2], location="bottom", aspect=32, shrink=0.8)

    def plot_tasks(self, axes=[0, 1]):
        s = 32

        gridded = self.n_dof == 2

        self.task_fig, self.task_axes = plt.subplots(
            self.num_tasks,
            3,
            figsize=(10, 4 * (self.num_tasks)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_2d(self.task_axes)

        self.task_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        for itask, task in enumerate(self.tasks):
            task_norm = mpl.colors.Normalize(*np.nanpercentile(task.regressor.Y, q=[1, 99]))

            self.task_axes[itask, 0].set_ylabel(task.name)

            self.task_axes[itask, 0].set_title("samples")
            self.task_axes[itask, 1].set_title("posterior mean")
            self.task_axes[itask, 2].set_title("posterior std. dev.")

            data_ax = self.task_axes[itask, 0].scatter(
                *task.regressor.X.T[axes], s=s, c=task.regressor.Y, norm=task_norm, cmap="plasma"
            )

            if gridded:
                self.task_axes[itask, 1].pcolormesh(
                    *self.dim_mids[axes],
                    task.regressor.mean(self.test_X_grid)[..., 0],
                    shading="nearest",
                    cmap="plasma",
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[itask, 2].pcolormesh(
                    *self.dim_mids[axes],
                    task.regressor.sigma(self.test_X_grid)[..., 0],
                    shading="nearest",
                    cmap="plasma",
                )

            else:
                self.task_axes[itask, 1].scatter(
                    *self.test_X.T[axes], s=s, c=task.regressor.mean(self.test_X), norm=task_norm, cmap="plasma"
                )
                sigma_ax = self.task_axes[itask, 2].scatter(
                    *self.test_X.T[axes], s=s, c=task.regressor.sigma(self.test_X), cmap="plasma"
                )

            self.task_fig.colorbar(data_ax, ax=self.task_axes[itask, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(sigma_ax, ax=self.task_axes[itask, 2], location="bottom", aspect=32, shrink=0.8)

    # talk to the model

    def fitness_estimate(self, X):
        return self.regressor.mean(X.reshape(-1, self.n_dof)).reshape(X.shape[:-1])

    def fitness_sigma(self, X):
        return self.regressor.sigma(X.reshape(-1, self.n_dof)).reshape(X.shape[:-1])

    def fitness_entropy(self, X):
        return np.log(np.sqrt(2 * np.pi * np.e) * self.fitness_sigma(X) + 1e-12)

    def validate(self, X):
        return self.classifier.p(X.reshape(-1, self.n_dof)).reshape(X.shape[:-1])

    def delay_estimate(self, X):
        return self.timer.mean(X.reshape(-1, self.n_dof)).reshape(X.shape[:-1])

    def delay_sigma(self, X):
        return self.timer.sigma(X.reshape(-1, self.n_dof)).reshape(X.shape[:-1])

    def _negative_A_optimality(self, X):
        """
        The negative trace of the inverse Fisher information matrix contingent on sampling the passed X.
        """
        test_X = X
        invFIM = np.linalg.inv(self.regressor._contingent_fisher_information_matrix(test_X, delta=1e-3))
        return np.array(list(map(np.trace, invFIM)))

    def _negative_D_optimality(self, X):
        """
        The negative determinant of the inverse Fisher information matrix contingent on sampling the passed X.
        """
        test_X = X
        FIM_stack = self.regressor._contingent_fisher_information_matrix(test_X, delta=1e-3)
        return -np.array(list(map(np.linalg.det, FIM_stack)))
