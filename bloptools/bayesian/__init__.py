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

COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen"]


def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


class Agent:
    def __init__(
        self,
        dofs,
        bounds,
        tasks,
        digestion,
        db,
        acquisition=None,
        detectors=None,
        initialization=None,
        training_iter=256,
        verbose=True,
    ):
        """
        A Bayesian optimization agent.

        Parameters
        ----------
        dofs : iterable of ophyd objects
            The degrees of freedom that the agent can control, which determine the output of the model.
        bounds : iterable of lower and upper bounds
            The bounds on each degree of freedom. This should be an array of shape (n_dof, 2).
        tasks : iterable of tasks
            The tasks which the agent will try to optimize.
        acquisition : Bluesky plan generator that takes arguments (dofs, inputs, dets)
            A plan that samples the beamline for some given inputs.
        digestion : function that takes arguments (db, uid)
            A function to digest the output of the acquisition.
        db : A databroker instance.
        """

        self.dofs = np.atleast_1d(dofs)
        self.bounds = np.atleast_2d(bounds)
        self.tasks = np.atleast_1d(tasks)
        self.initialization = initialization
        self.acquisition = acquisition if acquisition is not None else default_acquisition_plan
        self.digestion = digestion
        self.db = db

        for dof in self.dofs:
            dof.kind = "hinted"

        for i, task in enumerate(self.tasks):
            task.index = i

        self.dets = detectors if detectors is not None else np.array([])

        self.n_dof = len(self.dofs)
        self.target_names = [f"{task.name}_fitness" for task in self.tasks]
        self.n_tasks = len(self.tasks)

        self.training_iter = training_iter
        self.verbose = verbose

        MAX_TEST_POINTS = 2**11

        self.test_X = self.sampler(n=MAX_TEST_POINTS)

        self.test_inputs = self.unnormalize_inputs(self.test_X)

        n_per_dim = int(np.power(MAX_TEST_POINTS, 1 / self.n_dof))
        self.X_samples = np.linspace(0, 1, n_per_dim)
        self.test_X_grid = np.swapaxes(np.r_[np.meshgrid(*self.n_dof * [self.X_samples], indexing="ij")], 0, -1)
        self.test_inputs_grid = self.unnormalize_inputs(self.test_X_grid)

        self.inputs = np.empty((0, self.n_dof))
        self.targets = np.empty((0, self.n_tasks))
        self.table = pd.DataFrame()

        self._initialized = False

    def normalize_inputs(self, inputs):
        return (inputs - self.bounds.min(axis=1)) / self.bounds.ptp(axis=1)

    def unnormalize_inputs(self, X):
        return X * self.bounds.ptp(axis=1) + self.bounds.min(axis=1)

    def normalize_targets(self, targets):
        return (targets - self.targets_mean) / (1e-20 + self.targets_scale)

    def unnormalize_targets(self, targets):
        return targets * self.targets_scale + self.targets_mean

    @property
    def normalized_bounds(self):
        return (self.bounds - self.bounds.min(axis=1)[:, None]) / self.bounds.ptp(axis=1)[:, None]

    @property
    def targets_mean(self):
        return np.nanmean(self.targets, axis=0)

    @property
    def targets_scale(self):
        return np.nanstd(self.targets, axis=0)

    @property
    def normalized_targets(self):
        return self.normalize_targets(self.targets)

    @property
    def current_inputs(self):
        return np.array([dof.read()[dof.name]["value"] for dof in self.dofs])

    @property
    def dof_names(self):
        return [dof.name for dof in self.dofs]

    @property
    def det_names(self):
        return [det.name for det in self.dets]

    @property
    def best_sum_of_tasks(self):
        return np.nanmax(self.targets.sum(axis=1))

    @property
    def best_sum_of_tasks_inputs(self):
        return self.inputs[np.nanargmax(self.targets.sum(axis=1))]

    @property
    def go_to(self, inputs):
        yield from bps.mv(*[_ for items in zip(self.dofs, np.atleast_1d(inputs).T) for _ in items])

    @property
    def go_to_best_sum_of_tasks(self):
        yield from self.go_to(self.best_sum_of_tasks_inputs)

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

    def save(self, filepath="./agent_data.h5"):
        """
        Save the sampled inputs and targets of the agent to a file, which can be used
        to initialize a future agent.
        """
        with h5py.File(filepath, "w") as f:
            f.create_dataset("inputs", data=self.inputs)
        self.table.to_hdf(filepath, key="table")

    def forget(self, n):
        if n >= len(self.inputs):
            raise ValueError(f"Cannot forget last {n} points (the agent only has {len(self.inputs)} points).")
        self.tell(new_inputs=self.inputs[:-n], new_targets=self.targets[:-n], append=False)

    def sampler(self, n):
        """
        Returns $n$ quasi-randomly sampled points on the [0,1] ^ n_dof hypercube using Sobol sampling.
        """
        power_of_two = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        subset = np.random.choice(power_of_two, size=n, replace=False)
        return sp.stats.qmc.Sobol(d=self.n_dof, scramble=True).random(n=power_of_two)[subset]

    def initialize(
        self,
        filepath=None,
        init_scheme=None,
        n_init=4,
    ):
        """
        An initialization plan for the agent.
        This must be run before the agent can learn.
        It should be passed to a Bluesky RunEngine.
        """

        if filepath is not None:
            with h5py.File(filepath, "r") as f:
                init_inputs = f["inputs"][:]
            self.table = pd.read_hdf(filepath, key="table")
            init_targets = self.table.loc[:, self.target_names].values

        # experiment-specific stuff
        if self.initialization is not None:
            yield from self.initialization()

        # now let's get bayesian
        if init_scheme == "quasi-random":
            init_inputs = self.ask(strategy="quasi-random", n=n_init, route=True)
            init_table = yield from self.acquire(self.dofs, init_inputs, self.dets)
            self.table = pd.concat([self.table, init_table])
            self.table.index = np.arange(len(self.table))
            init_targets = init_table.loc[:, self.target_names].values

        else:
            raise Exception(
                "Could not initialize model! Either pass initial X and data, or specify one of:"
                "['quasi-random']."
            )

        if (init_inputs is None) or (init_targets is None):
            raise RuntimeError("Unhandled initialization error.")

        no_good_samples_tasks = np.isnan(init_targets).all(axis=0)
        if no_good_samples_tasks.any():
            raise ValueError(
                f"The tasks {[self.tasks[i].name for i in np.where(no_good_samples_tasks)[0]]} "
                f"don't have any good samples."
            )

        self.tell(new_inputs=init_inputs, new_targets=init_targets, reuse_hypers=True, verbose=self.verbose)

        self._initialized = True

    def tell(self, new_inputs, new_targets, append=True, **kwargs):
        """
        Inform the agent about new inputs and targets for the model.
        """
        if append:
            self.inputs = np.r_[self.inputs, np.atleast_2d(new_inputs)]
            self.targets = np.r_[self.targets, np.atleast_2d(new_targets)]
        else:
            self.inputs = new_inputs
            self.targets = new_targets

        self.X = self.normalize_inputs(self.inputs)

        # if hasattr(self.experiment, "IMAGE_NAME"):
        #     self.images = np.array([im for im in self.table[self.experiment.IMAGE_NAME].values])

        self.all_targets_valid = ~np.isnan(self.targets).any(axis=1)

        for task in self.tasks:
            task.targets = self.targets[:, task.index]
            task.normalized_targets = self.normalized_targets[:, task.index]

            task.targets_mean = np.nanmean(task.targets)
            task.targets_scale = np.nanstd(task.targets)

            task.classes = self.all_targets_valid.astype(int)

            regressor_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(task.MIN_NOISE_LEVEL).square(),
                    torch.tensor(task.MAX_NOISE_LEVEL).square(),
                ),
            ).double()

            task.regressor = models.BoTorchSingleTaskGP(
                train_inputs=torch.tensor(self.X[task.classes == 1]).double(),
                train_targets=torch.tensor(self.normalized_targets[:, task.index][task.classes == 1]).double(),
                likelihood=regressor_likelihood,
            ).double()

            task.regressor_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                task.regressor.likelihood, task.regressor
            )
            botorch.fit.fit_gpytorch_mll(task.regressor_mll, **kwargs)

        self.scalarization = botorch.acquisition.objective.ScalarizedPosteriorTransform(
            weights=torch.tensor(self.targets_scale).double(),
            offset=self.targets_mean.sum(),
        )

        dirichlet_classifier_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(self.all_targets_valid).long(), learn_additional_noise=True
        ).double()

        self.dirichlet_classifier = models.BoTorchDirichletClassifier(
            train_inputs=torch.tensor(self.X).double(),
            train_targets=dirichlet_classifier_likelihood.transformed_targets,
            likelihood=dirichlet_classifier_likelihood,
        ).double()

        self.dirichlet_classifier_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.dirichlet_classifier.likelihood, self.dirichlet_classifier
        )
        botorch.fit.fit_gpytorch_mll(self.dirichlet_classifier_mll, **kwargs)

        self.classifier = botorch.models.deterministic.GenericDeterministicModel(
            f=lambda X: self.dirichlet_classifier.log_prob(X)
        )

        self.multimodel = botorch.models.model.ModelList(*[task.regressor for task in self.tasks])

    def sample_acqf(self, acqf, n_test=256, optimize=False):
        """
        Return the point that optimizes a given acqusition function.
        """

        def acq_loss(x, *args):
            return -acqf(x, *args).detach().numpy()

        acq_args = (self,)

        test_X = self.sampler(n=n_test)
        init_X = test_X[acq_loss(test_X, *acq_args).argmin()]

        if optimize:
            res = sp.optimize.minimize(
                fun=acq_loss,
                args=acq_args,
                x0=init_X,
                bounds=self.normalized_bounds,
                method="SLSQP",
                options={"maxiter": 256},
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
        The next $n$ points to sample, recommended by the agent.
        """

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

        if strategy.lower() == "quasi-random":
            return self.unnormalize_inputs(self.sampler(n=n))

        if not self._initialized:
            raise RuntimeError("The agent is not initialized!")

        if tasks is None:
            tasks = self.tasks

        if classifier is None:
            classifier = self.classifier

        if (not greedy) or (n == 1):
            acqf = None

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

            self.forget(n=n)  # forget what you saw here

            return inputs_to_sample

    def acquire(self, dofs, inputs, dets):
        """
        Acquire and digest according to the agent's acquisition and digestion plans.

        This should yield a table of sampled tasks with the same length as the sampled inputs.
        """
        try:
            uid = yield from self.acquisition(dofs, inputs, dets)
            products = self.digestion(self.db, uid)
            acq_table = pd.DataFrame(inputs, columns=[dof.name for dof in dofs])
            acq_table.insert(0, "timestamp", pd.Timestamp.now())

            for key, values in products.items():
                acq_table.loc[:, key] = values

            # compute the fitness for each task
            for index, entry in acq_table.iterrows():
                for task in self.tasks:
                    acq_table.loc[index, f"{task.name}_fitness"] = task.get_fitness(entry)

        except Exception as err:
            acq_table = pd.DataFrame()
            logging.warning(repr(err))

        if not len(inputs) == len(acq_table):
            raise ValueError("The resulting table must be the same length as the sampled inputs!")

        return acq_table

    def learn(
        self, strategy, n_iter=1, n_per_iter=1, reuse_hypers=True, upsample=1, verbose=True, plots=[], **kwargs
    ):
        """
        This iterates the learning algorithm, looping over ask -> acquire -> tell.
        It should be passed to a Bluesky RunEngine.
        """

        print(f'learning with strategy "{strategy}" ...')

        for i in range(n_iter):
            inputs_to_sample = np.atleast_2d(self.ask(n=n_per_iter, strategy=strategy, **kwargs))

            new_table = yield from self.acquire(self.dofs, inputs_to_sample, self.dets)

            self.table = pd.concat([self.table, new_table])
            self.table.index = np.arange(len(self.table))
            new_targets = new_table.loc[:, self.target_names].values

            self.tell(new_inputs=inputs_to_sample, new_targets=new_targets, reuse_hypers=reuse_hypers)

    def plot_tasks(self, **kwargs):
        if self.n_dof == 1:
            self._plot_tasks_one_dof(**kwargs)

        else:
            self._plot_tasks_many_dofs(**kwargs)

    def plot_constraints(self, **kwargs):
        if self.n_dof == 1:
            self._plot_constraints_one_dof(**kwargs)

        else:
            self._plot_constraints_many_dofs(**kwargs)

    def _plot_constraints_one_dof(self, size=32):
        self.class_fig, self.class_ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, constrained_layout=True)

        self.class_ax.scatter(self.inputs, self.all_targets_valid.astype(int), s=size)

        x = torch.tensor(self.test_X_grid.reshape(-1, self.n_dof)).double()
        log_prob = self.dirichlet_classifier.log_prob(x).detach().numpy().reshape(self.test_X_grid.shape[:-1])

        self.class_ax.plot(self.test_inputs_grid.ravel(), np.exp(log_prob))

        self.class_ax.set_xlim(*self.bounds[0])

    def _plot_constraints_many_dofs(self, axes=[0, 1], shading="nearest", cmap="inferno", size=32, gridded=None):
        if gridded is None:
            gridded = self.n_dof == 2

        self.class_fig, self.class_axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True
        )

        for ax in self.class_axes.ravel():
            ax.set_xlabel(self.dofs[axes[0]].name)
            ax.set_ylabel(self.dofs[axes[1]].name)

        data_ax = self.class_axes[0].scatter(
            *self.inputs.T[:2], s=size, c=self.all_targets_valid.astype(int), vmin=0, vmax=1, cmap=cmap
        )

        if gridded:
            x = torch.tensor(self.test_X_grid.reshape(-1, self.n_dof)).double()
            log_prob = self.dirichlet_classifier.log_prob(x).detach().numpy().reshape(self.test_X_grid.shape[:-1])

            self.class_axes[1].pcolormesh(
                *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                np.exp(log_prob),
                shading=shading,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

        else:
            x = torch.tensor(self.test_X).double()
            log_prob = self.dirichlet_classifier.log_prob(x).detach().numpy()

            self.class_axes[1].scatter(*self.test_X.T[axes], s=size, c=np.exp(log_prob), vmin=0, vmax=1, cmap=cmap)

        self.class_fig.colorbar(data_ax, ax=self.class_axes[:2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self.bounds[axes[0]])
            ax.set_ylim(*self.bounds[axes[1]])

    def _plot_tasks_one_dof(self, size=32, lw=1e0):
        self.task_fig, self.task_axes = plt.subplots(
            self.n_tasks,
            1,
            figsize=(6, 4 * self.n_tasks),
            sharex=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_1d(self.task_axes)

        for itask, task in enumerate(self.tasks):
            color = COLOR_LIST[itask]

            self.task_axes[itask].set_ylabel(task.name)

            x = torch.tensor(self.test_X_grid).double()
            task_posterior = task.regressor.posterior(x)
            task_mean = task_posterior.mean.detach().numpy() * task.targets_scale + task.targets_mean
            task_sigma = task_posterior.variance.sqrt().detach().numpy() * task.targets_scale

            self.task_axes[itask].scatter(self.inputs, task.targets, s=size, color=color)
            self.task_axes[itask].plot(self.test_inputs_grid.ravel(), task_mean, lw=lw, color=color)

            for z in [1, 2]:
                self.task_axes[itask].fill_between(
                    self.test_inputs_grid.ravel(),
                    (task_mean - z * task_sigma).ravel(),
                    (task_mean + z * task_sigma).ravel(),
                    lw=lw,
                    color=color,
                    alpha=0.5**z,
                )

            self.task_axes[itask].set_xlim(*self.bounds[0])

    def _plot_tasks_many_dofs(self, axes=[0, 1], shading="nearest", cmap="inferno", gridded=None, size=32):
        if gridded is None:
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
                *self.inputs.T[axes], s=size, c=task.targets, norm=task_norm, cmap=cmap
            )

            if gridded:
                x = torch.tensor(self.test_X_grid).double()
            else:
                x = torch.tensor(self.test_X).double()

            task_posterior = task.regressor.posterior(x)
            task_mean = task_posterior.mean.detach().numpy() * task.targets_scale + task.targets_mean
            task_sigma = task_posterior.variance.sqrt().detach().numpy() * task.targets_scale

            if gridded:
                self.task_axes[itask, 1].pcolormesh(
                    *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                    task_mean.reshape(self.test_X_grid.shape[:-1]),
                    shading=shading,
                    cmap=cmap,
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[itask, 2].pcolormesh(
                    *(self.bounds[axes].ptp(axis=1) * self.X_samples[:, None] + self.bounds[axes].min(axis=1)).T,
                    task_sigma.reshape(self.test_X_grid.shape[:-1]),
                    shading=shading,
                    cmap=cmap,
                )

            else:
                self.task_axes[itask, 1].scatter(
                    *self.test_inputs.T[axes], s=size, c=task_mean, norm=task_norm, cmap=cmap
                )
                sigma_ax = self.task_axes[itask, 2].scatter(
                    *self.test_inputs.T[axes], s=size, c=task_sigma, cmap=cmap
                )

            self.task_fig.colorbar(data_ax, ax=self.task_axes[itask, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(sigma_ax, ax=self.task_axes[itask, 2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self.bounds[axes[0]])
            ax.set_ylim(*self.bounds[axes[1]])
