import copy
import logging

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401
import botorch
import gpytorch
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import torch
from matplotlib import pyplot as plt

from .. import utils
from . import acquisition, models

mpl.rc("image", cmap="coolwarm")


class Agent:
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

        self.target_names = [f"{task.name}_fitness" for task in tasks]

        self.n_tasks = len(tasks)

        self._initialized = False

        self.db = db
        self.training_iter = training_iter

        self.verbose = verbose

        MAX_TEST_POINTS = 2**10

        self.test_X = sp.stats.qmc.Sobol(d=self.n_dof, scramble=True).random(n=MAX_TEST_POINTS)
        self.test_inputs = self.unnormalize_inputs(self.test_X)

        n_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
        self.X_samples = np.linspace(0, 1, n_per_dim)
        self.test_X_grid = np.swapaxes(np.r_[np.meshgrid(*self.n_dof * [self.X_samples], indexing="ij")], 0, -1)
        self.test_inputs_grid = self.unnormalize_inputs(self.test_X)

        self.inputs = np.empty((0, self.n_dof))
        self.targets = np.empty((0, self.n_tasks))
        self.table = pd.DataFrame()

    def normalize_inputs(self, inputs):
        return (inputs - self.bounds.min(axis=1)) / self.bounds.ptp(axis=1)

    def unnormalize_inputs(self, X):
        return X * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

    def normalize_targets(self, targets):
        return (targets - np.nanmean(self.targets, axis=0)) / np.nanstd(self.targets, axis=0)

    def unnormalize_targets(self, targets):
        return targets * np.nanstd(self.targets, axis=0) + np.nanmean(self.targets, axis=0)

    def measurement_plan(self):
        yield from bp.count(detectors=self.dets)

    def unpack_run(self):
        return None

    # def load(filepath, **kwargs):
    # with h5py.File(filepath, "r") as f:
    #     X = f["X"][:]
    # table = pd.read_hdf(filepath, key="table")
    # return BayesianOptimizationAgent(init_X=X, init_table=table, **kwargs)

    def initialize(
        self,
        filepath=None,
        init_scheme=None,
        n_init=4,
    ):
        if filepath is not None:
            with h5py.File(filepath, "r") as f:
                inputs = f["inputs"][:]
            self.table = pd.read_hdf(filepath, key="table")
            targets = self.table.loc[:, self.target_names].values
            self.tell(new_inputs=inputs, new_targets=targets)
            return

        # experiment-specific stuff
        yield from self.experiment.initialize()

        # now let's get bayesian
        if init_scheme == "quasi-random":
            unrouted_inputs = self.unnormalize_inputs(
                sp.stats.qmc.Sobol(d=self.n_dof, scramble=True).random(n=n_init)
            )
            routing_index, _ = utils.get_routing(self.current_inputs, unrouted_inputs)
            init_inputs = unrouted_inputs[routing_index]
            self.table = yield from self.acquire_with_bluesky(init_inputs)
            init_targets = self.table.loc[:, self.target_names].values

        else:
            raise Exception(
                "Could not initialize model! Either pass initial X and data, or specify one of:"
                "['quasi-random']."
            )

        if (init_inputs is None) or (init_targets is None):
            raise RuntimeError()

        self.tell(new_inputs=init_inputs, new_targets=init_targets, reuse_hypers=True, verbose=self.verbose)

        self._initialized = True

    @property
    def current_inputs(self):
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

        x_min, x_max, y_min, y_max, width_x, width_y = self.table.loc[
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
            f.create_dataset("inputs", data=self.inputs)
        self.table.to_hdf(filepath, key="table")

    def untell(self, n):
        self.inputs = self.inputs[:-n]
        self.targets = self.targets[:-n]

        for task in self.tasks:
            task.regressor.set_train_data(
                task.regressor.train_inputs[0][:-n], task.regressor.train_targets[:-n], strict=False
            )

    def tell(self, new_inputs, new_targets, **kwargs):
        self.inputs = np.r_[self.inputs, np.atleast_2d(new_inputs)]
        self.X = self.normalize_inputs(self.inputs)

        self.targets = np.r_[self.targets, np.atleast_2d(new_targets)]
        self.normalized_targets = self.normalize_targets(self.targets)

        if hasattr(self.experiment, "IMAGE_NAME"):
            self.images = np.array([im for im in self.table[self.experiment.IMAGE_NAME].values])

        all_targets_valid = ~np.isnan(self.targets).any(axis=1)

        for itask, task in enumerate(self.tasks):
            task.targets = self.targets[:, itask]
            task.normalized_targets = self.normalized_targets[:, itask]

            task.targets_mean = np.nanmean(task.targets)
            task.targets_scale = np.nanstd(task.targets)
            task.classes = (~np.isnan(task.targets)).astype(int)

            regressor_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(task.MIN_NOISE_LEVEL).square(),
                    torch.tensor(task.MAX_NOISE_LEVEL).square(),
                ),
            ).double()

            task.regressor = models.BoTorchSingleTaskGP(
                train_inputs=torch.tensor(self.X[task.classes == 1]).double(),
                train_targets=torch.tensor(task.normalized_targets[task.classes == 1]).double(),
                likelihood=regressor_likelihood,
            ).double()

            task.regressor_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                task.regressor.likelihood, task.regressor
            )
            botorch.fit.fit_gpytorch_mll(task.regressor_mll, **kwargs)

        classifier_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(all_targets_valid).long(), learn_additional_noise=True
        ).double()

        self.classifier = models.BoTorchClassifier(
            train_inputs=torch.tensor(self.X).double(),
            train_targets=classifier_likelihood.transformed_targets,
            likelihood=classifier_likelihood,
        ).double()

        self.classifier_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.classifier.likelihood, self.classifier)
        botorch.fit.fit_gpytorch_mll(self.classifier_mll, **kwargs)

        self.multimodel = botorch.models.model.ModelList(*[task.regressor for task in self.tasks])

        # self.regressor.set_data(X, Y)
        # self.regressor.train(step_limit=self.training_iter)

        # c = (~np.isnan(Y).any(axis=-1)).astype(int)
        # self.classifier.set_data(X, c)
        # self.classifier.train(step_limit=self.training_iter)

    def acquire_with_bluesky(self, X, verbose=False):
        if verbose:
            print(f"sampling {X}")

        try:
            uid = yield from bp.list_scan(
                self.dets, *[_ for items in zip(self.dofs, np.atleast_2d(X).T) for _ in items]
            )

            new_table = self.db[uid].table(fill=True)
            new_table.loc[:, "uid"] = uid

        except Exception as err:
            new_table = pd.DataFrame()
            logging.warning(repr(err))

        for index in new_table.index:
            for k, v in self.experiment.postprocess(new_table.loc[index]).items():
                new_table.loc[index, k] = v
            for task in self.tasks:
                new_table.loc[index, f"{task.name}_fitness"] = task.get_fitness(new_table.loc[index])

        self.table = pd.concat([self.table, new_table])
        self.table.index = np.arange(len(self.table))

        return new_table

    def sample_acqf(self, acqf, n_test=2048, optimize=False):
        def acq_loss(x, *args):
            return -acqf(x, *args)

        acq_args = (self,)

        test_X = sp.stats.qmc.Sobol(d=self.n_dof, scramble=True).random(n=n_test)
        init_X = test_X[acq_loss(test_X, *acq_args).argmin()]

        # print(init_X)

        if optimize:
            res = sp.optimize.minimize(
                fun=acq_loss, args=acq_args, x0=init_X, bounds=self.bounds, method="SLSQP", options={"maxiter": 32}
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
        normalize=False,
    ):
        """
        Recommends the next $n$ points to sample.
        """

        if not self._initialized:
            raise RuntimeError("The agent is not initialized!")

        if route:
            unrouted_points = self.ask(
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

            routing_index, _ = utils.get_routing(self.current_inputs, unrouted_points)
            return unrouted_points[routing_index]

        if tasks is None:
            tasks = self.tasks

        if classifier is None:
            classifier = self.classifier

        if (not greedy) or (n == 1):
            acqf = None

            if strategy.lower() == "est":  # maximize the expected improvement
                acqf = acquisition.expected_sum_of_tasks

            if strategy.lower() == "esti":  # maximize the expected improvement
                acqf = acquisition.log_expected_sum_of_tasks_improvement

            if acqf is None:
                raise ValueError(f'Unrecognized strategy "{strategy}".')

            X, loss = self.sample_acqf(acqf, optimize=optimize)

            return self.unnormalize_inputs(np.atleast_2d(X))

        if greedy and (n > 1):
            dummy_tasks = [copy.deepcopy(task) for task in self.tasks]

            inputs_to_sample = np.zeros((0, self.n_dof))

            for i in range(n):
                new_input = self.ask(strategy=strategy, tasks=dummy_tasks, classifier=classifier, greedy=True, n=1)

                inputs_to_sample = np.r_[inputs_to_sample, new_input]

                if not (i + 1 == n):  # no point if we're on the last iteration
                    fantasy_multimodel = botorch.models.model.ModelList(*[task.regressor for task in self.tasks])
                    fantasy_posterior = fantasy_multimodel.posterior(
                        torch.tensor(self.normalize_inputs(new_input)).double()
                    )

                    fantasy_targets = fantasy_posterior.mean - disappointment * fantasy_posterior.variance.sqrt()
                    new_targets = self.unnormalize_targets(fantasy_targets.detach().numpy())

                    self.tell(new_inputs=new_input, new_targets=new_targets)

            self.untell(n=n)  # forget what you saw here

            return inputs_to_sample

    def learn(
        self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
    ):
        # ip.display.clear_output(wait=True)
        print(f'learning with strategy "{strategy}" ...')

        for i in range(n_iter):
            inputs_to_sample = np.atleast_2d(
                self.ask(n=n_per_iter, strategy=strategy, **kwargs)
            )  # get point(s) to sample from the strategizer

            # n_original = len(X_to_sample)
            # n_upsample = upsample * n_original + 1

            # upsampled_X_to_sample = sp.interpolate.interp1d(
            #     np.arange(n_original + 1), np.r_[self.current_X[None], X_to_sample], axis=0
            # )(np.linspace(0, n_original, n_upsample)[1:])

            new_table = yield from self.acquire_with_bluesky(inputs_to_sample)
            new_targets = new_table.loc[:, self.target_names].values

            self.tell(new_inputs=inputs_to_sample, new_targets=new_targets, reuse_hypers=reuse_hypers)

            # if "state" in plots:
            #     self.plot_state(remake=False, gridded=self.gridded_plots)
            # if "fitness" in plots:
            #     self.plot_fitness(remake=False)

            # if verbose:
            #     n_X = len(sampled_X)
            #     df_to_print = pd.DataFrame(
            #         np.c_[self.inputs, self.experiment.HINTED_STATS],
            #         columns=[*self.dof_names, *self.experiment.HINTED_STATS]
            #     ).iloc[-n_X:]
            #     print(df_to_print)

    def plot_constraints(self, axes=[0, 1]):
        s = 32

        gridded = self.n_dof == 2

        self.class_fig, self.class_axes = plt.subplots(
            1, 3, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True
        )

        for ax in self.class_axes.ravel():
            ax.set_xlabel(self.dofs[axes[0]].name)
            ax.set_ylabel(self.dofs[axes[1]].name)

        data_ax = self.class_axes[0].scatter(
            *self.inputs.T[:2], s=s, c=~np.isnan(self.targets).any(axis=1), vmin=0, vmax=1, cmap="plasma"
        )

        if gridded:
            x = torch.tensor(self.test_X_grid.reshape(-1, self.n_dof)).double()
            log_prob = self.classifier.log_prob(x).detach().numpy().reshape(self.test_X_grid.shape[:-1])
            entropy = -log_prob * np.exp(log_prob) - (1 - log_prob) * np.exp(1 - log_prob)

            self.class_axes[1].pcolormesh(
                *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                np.exp(log_prob),
                shading="nearest",
                cmap="plasma",
                vmin=0,
                vmax=1,
            )
            entropy_ax = self.class_axes[2].pcolormesh(
                *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                entropy,
                shading="nearest",
                cmap="plasma",
            )

        else:
            x = torch.tensor(self.test_X).double()
            log_prob = self.classifier.log_prob(x).detach().numpy()
            entropy = -log_prob * np.exp(log_prob) - (1 - log_prob) * np.exp(1 - log_prob)

            self.class_axes[1].scatter(
                *self.test_X.T[axes], s=s, c=np.exp(log_prob), vmin=0, vmax=1, cmap="plasma"
            )
            entropy_ax = self.class_axes[2].scatter(*self.test_X.T[axes], s=s, c=entropy, cmap="plasma")

        self.class_fig.colorbar(data_ax, ax=self.class_axes[:2], location="bottom", aspect=32, shrink=0.8)
        self.class_fig.colorbar(entropy_ax, ax=self.class_axes[2], location="bottom", aspect=32, shrink=0.8)

    def plot_tasks(self, axes=[0, 1]):
        s = 32

        gridded = self.n_dof == 2

        self.task_fig, self.task_axes = plt.subplots(
            self.n_tasks,
            3,
            figsize=(10, 4 * self.n_tasks),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_2d(self.task_axes)

        self.task_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        for itask, task in enumerate(self.tasks):
            task_norm = mpl.colors.Normalize(*np.nanpercentile(task.targets, q=[1, 99]))

            self.task_axes[itask, 0].set_ylabel(task.name)

            self.task_axes[itask, 0].set_title("samples")
            self.task_axes[itask, 1].set_title("posterior mean")
            self.task_axes[itask, 2].set_title("posterior std. dev.")

            data_ax = self.task_axes[itask, 0].scatter(
                *self.inputs.T[axes], s=s, c=task.targets, norm=task_norm, cmap="plasma"
            )

            if gridded:
                x = torch.tensor(self.test_X_grid.reshape(-1, self.n_dof)).double()
                task_posterior = task.regressor.posterior(x)
                task_mean = task_posterior.mean.detach().numpy() * task.targets_scale + task.targets_mean
                task_sigma = task_posterior.variance.sqrt().detach().numpy() * task.targets_scale

                self.task_axes[itask, 1].pcolormesh(
                    *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                    task_mean.reshape(self.test_X_grid.shape[:-1]),
                    shading="nearest",
                    cmap="plasma",
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[itask, 2].pcolormesh(
                    *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                    task_sigma.reshape(self.test_X_grid.shape[:-1]),
                    shading="nearest",
                    cmap="plasma",
                )

            else:
                x = torch.tensor(self.test_X).double()
                task_posterior = task.regressor.posterior(x)
                task_mean = task_posterior.mean.detach().numpy() * task.targets_scale + task.targets_mean
                task_sigma = task_posterior.variance.sqrt().detach().numpy() * task.targets_scale

                self.task_axes[itask, 1].scatter(
                    *self.test_inputs.T[axes], s=s, c=task_mean, norm=task_norm, cmap="plasma"
                )
                sigma_ax = self.task_axes[itask, 2].scatter(
                    *self.test_inputs.T[axes], s=s, c=task_sigma, cmap="plasma"
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
