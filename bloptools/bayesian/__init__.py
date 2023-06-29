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
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement
from matplotlib import pyplot as plt

from .. import utils
from . import acquisition, models

mpl.rc("image", cmap="coolwarm")

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen"]
DEFAULT_COLORMAP = "inferno"


def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


def default_digestion_plan(db, uid):
    return db[uid].table()


# let's be specific about our terminology.
#
# dofs: degrees of freedom are things that can change
# inputs: these are the values of the dofs, which may be transformed/normalized
# targets: these are what our model tries to predict from the inputs
# tasks: these are quantities that our agent will try to optimize over


class Agent:
    MAX_TEST_POINTS = 2**10

    def __init__(
        self,
        active_dofs,
        active_dof_bounds,
        tasks,
        db,
        **kwargs,
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

        self.active_dofs = np.atleast_1d(active_dofs)
        self.active_dof_bounds = np.atleast_2d(active_dof_bounds)
        self.tasks = np.atleast_1d(tasks)
        self.db = db

        self.verbose = kwargs.get("verbose", False)
        self.ignore_acquisition_errors = kwargs.get("ignore_acquisition_errors", False)

        self.initialization = kwargs.get("initialization", None)
        self.acquisition_plan = kwargs.get("acquisition_plan", default_acquisition_plan)
        self.digestion = kwargs.get("digestion", default_digestion_plan)

        self.acquisition = acquisition.Acquisition()

        self.dets = np.atleast_1d(kwargs.get("detectors", []))
        self.passive_dofs = np.atleast_1d(kwargs.get("passive_dofs", []))

        for i, task in enumerate(self.tasks):
            task.index = i

        self.n_active_dof = self.active_dofs.size
        self.n_passive_dof = self.passive_dofs.size

        self.dofs = np.r_[self.active_dofs, self.passive_dofs]

        self.n_dof = self.n_active_dof + self.n_passive_dof

        self.n_tasks = len(self.tasks)

        self.training_iter = kwargs.get("training_iter", 256)

        # self.test_normalized_active_inputs = self.sampler(n=self.MAX_TEST_POINTS)
        self.test_active_inputs = self.unnormalize_active_inputs(self.sampler(n=self.MAX_TEST_POINTS))

        n_per_active_dim = int(np.power(self.MAX_TEST_POINTS, 1 / self.n_active_dof))

        test_normalized_active_inputs_grid = np.swapaxes(
            np.r_[np.meshgrid(*self.n_active_dof * [np.linspace(0, 1, n_per_active_dim)])], 0, -1
        )

        self.test_active_inputs_grid = self.unnormalize_active_inputs(test_normalized_active_inputs_grid)

        self.table = pd.DataFrame()

        self._initialized = False

    @property
    def input_transform(self):
        coefficient = torch.tensor(self.inputs.values.ptp(axis=0))
        offset = torch.tensor(self.inputs.values.min(axis=0))
        return botorch.models.transforms.input.AffineInputTransform(
            d=self.n_dof, coefficient=coefficient, offset=offset
        )

    def save(self, filepath="./agent_data.h5"):
        """
        Save the sampled inputs and targets of the agent to a file, which can be used
        to initialize a future agent.
        """
        with h5py.File(filepath, "w") as f:
            f.create_dataset("inputs", data=self.inputs)
        self.table.to_hdf(filepath, key="table")

    def forget(self, index):
        self.tell(new_table=self.table.drop(index=index), append=False)

    def sampler(self, n):
        """
        Returns $n$ quasi-randomly sampled points on the [0,1] ^ n_active_dof hypercube using Sobol sampling.
        """
        min_power_of_two = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        subset = np.random.choice(min_power_of_two, size=n, replace=False)
        return sp.stats.qmc.Sobol(d=self.n_active_dof, scramble=True).random(n=min_power_of_two)[subset]

    def active_dof_sampler(self, n, q=1):
        return botorch.utils.sampling.draw_sobol_samples(torch.tensor(self.active_dof_bounds.T), n=n, q=q)

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

        init_table = None

        if filepath is not None:
            init_table = pd.read_hdf(filepath, key="table")

        # experiment-specific stuff
        if self.initialization is not None:
            yield from self.initialization()

        # now let's get bayesian
        if init_scheme == "quasi-random":
            init_dof_inputs = self.ask(strategy="quasi-random", n=n_init, route=True)
            init_table = yield from self.acquire(dof_inputs=init_dof_inputs)

        else:
            raise Exception(
                "Could not initialize model! Either pass initial X and data, or specify one of:"
                "['quasi-random']."
            )

        if init_table is None:
            raise RuntimeError("Unhandled initialization error.")

        no_good_samples_tasks = np.isnan(init_table.loc[:, self.target_names]).all(axis=0)
        if no_good_samples_tasks.any():
            raise ValueError(
                f"The tasks {[self.tasks[i].name for i in np.where(no_good_samples_tasks)[0]]} "
                f"don't have any good samples."
            )

        self.tell(new_table=init_table, verbose=self.verbose)
        self._initialized = True

    def tell(self, new_table=None, append=True, **kwargs):
        """
        Inform the agent about new inputs and targets for the model.
        """

        new_table = pd.DataFrame() if new_table is None else new_table

        self.table = pd.concat([self.table, new_table]) if append else new_table

        self.table.loc[:, "total_fitness"] = self.table.loc[:, self.task_names].fillna(-np.inf).sum(axis=1)
        self.table.index = np.arange(len(self.table))

        # self.normalized_inputs = self.normalize_inputs(self.inputs)

        self.all_targets_valid = ~np.isnan(self.targets).any(axis=1)

        for task in self.tasks:
            task.targets = self.targets.loc[:, task.name]
            #
            # task.targets_mean = np.nanmean(task.targets, axis=0)
            # task.targets_scale = np.nanstd(task.targets, axis=0)

            # task.normalized_targets = self.normalized_targets.loc[:, task.name]

            task.feasibility = self.all_targets_valid.astype(int)

            train_inputs = torch.tensor(self.inputs.loc[task.feasibility == 1].values).double().unsqueeze(0)
            train_targets = (
                torch.tensor(task.targets.loc[task.feasibility == 1].values).double().unsqueeze(0).unsqueeze(-1)
            )

            if train_inputs.ndim == 1:
                train_inputs = train_inputs.unsqueeze(-1)
            if train_targets.ndim == 1:
                train_targets = train_targets.unsqueeze(-1)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(task.MIN_NOISE_LEVEL).square(),
                    torch.tensor(task.MAX_NOISE_LEVEL).square(),
                ),
            ).double()

            task.regressor = models.BoTorchSingleTaskGP(
                train_inputs=train_inputs,
                train_targets=train_targets,
                likelihood=likelihood,
                input_transform=self.input_transform,
                outcome_transform=botorch.models.transforms.outcome.Standardize(m=1, batch_shape=torch.Size((1,))),
            ).double()

            task.regressor_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                task.regressor.likelihood, task.regressor
            )
            botorch.fit.fit_gpytorch_mll(task.regressor_mll, **kwargs)

        self.task_scalarization = botorch.acquisition.objective.ScalarizedPosteriorTransform(
            weights=torch.tensor([*[task.weight for task in self.tasks], 1]).double(),
            offset=0,
        )

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(self.all_targets_valid).long(), learn_additional_noise=True
        ).double()

        self.dirichlet_classifier = models.BoTorchDirichletClassifier(
            train_inputs=torch.tensor(self.inputs.values).double(),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            likelihood=dirichlet_likelihood,
            input_transform=self.input_transform,
        ).double()

        self.dirichlet_classifier_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.dirichlet_classifier.likelihood, self.dirichlet_classifier
        )
        botorch.fit.fit_gpytorch_mll(self.dirichlet_classifier_mll, **kwargs)

        self.feas_model = botorch.models.deterministic.GenericDeterministicModel(
            f=lambda X: self.dirichlet_classifier.log_prob(X)
        )

        self.targets_model = botorch.models.model_list_gp_regression.ModelListGP(
            *[task.regressor for task in self.tasks]
        )

        self.task_model = botorch.models.model.ModelList(*[task.regressor for task in self.tasks], self.feas_model)

    # def sample_acqf(self, acqf, n_test=256, optimize=False):
    #     """
    #     Return the point that optimizes a given acqusition function.
    #     """

    #     def acq_loss(x, *args):
    #         return -acqf(x, *args)

    #     acq_args = (self,)

    #     test_inputs = self.unnormalize_active_inputs(self.sampler(n=n_test))
    #     init_inputs = test_inputs[acq_loss(test_inputs, *acq_args).argmin()]

    #     if optimize:
    #         res = sp.optimize.minimize(
    #             fun=acq_loss,
    #             args=acq_args,
    #             x0=init_inputs,
    #             bounds=self.active_dof_bounds,
    #             method="SLSQP",
    #             options={"maxiter": 256},
    #         )
    #         acq_inputs = res.x
    #     else:
    #         acq_inputs = init_inputs

    #     return acq_inputs, acq_loss(acq_inputs, *acq_args)

    # def ask(
    #     self,
    #     tasks=None,
    #     classifier=None,
    #     strategy=None,
    #     greedy=True,
    #     n=1,
    #     disappointment=0,
    #     route=True,
    #     cost_model=None,
    #     n_test=1024,
    #     optimize=True,
    # ):
    #     """
    #     The next $n$ points to sample, recommended by the agent.
    #     """

    #     if route:
    #         unrouted_points = self.ask(
    #             tasks=tasks,
    #             classifier=classifier,
    #             strategy=strategy,
    #             greedy=greedy,
    #             n=n,
    #             disappointment=disappointment,
    #             route=False,
    #             cost_model=cost_model,
    #             n_test=n_test,
    #         )

    #         routing_index, _ = utils.get_routing(self.read_active_dofs, unrouted_points)
    #         return unrouted_points[routing_index]

    #     if strategy.lower() == "quasi-random":
    #         return self.unnormalize_active_inputs(self.sampler(n=n))

    #     if not self._initialized:
    #         raise RuntimeError("The agent is not initialized!")

    #     if tasks is None:
    #         tasks = self.tasks

    #     if classifier is None:
    #         classifier = self.feas_model

    #     if (not greedy) or (n == 1):
    #         acqf = None

    #         if strategy.lower() == "ei":  # maximize the expected improvement

    #             acq_func = LogExpectedImprovement(agent.task_model,
    #                                         best_f=agent.best_sum_of_tasks,
    #                                         posterior_transform=agent.scalarization)

    #             acqf = self.acquisition.log_expected_improvement

    #         if strategy.lower() == "pi":  # maximize the probability improvement
    #             acqf = self.acquisition.log_probability_of_improvement

    #         if acqf is None:
    #             raise ValueError(f'Unrecognized strategy "{strategy}".')

    #         inputs_to_sample, loss = self.sample_acqf(acqf, optimize=optimize)

    #         return np.atleast_2d(inputs_to_sample)

    #     if greedy and (n > 1):
    #         dummy_tasks = [copy.deepcopy(task) for task in self.tasks]

    #         inputs_to_sample = np.zeros((0, self.n_dof))

    #         for i in range(n):

    #             new_input = self.ask(strategy=strategy, tasks=dummy_tasks, classifier=classifier, greedy=True, n=1)

    #             inputs_to_sample = np.r_[inputs_to_sample, new_input]

    #             if not (i + 1 == n):  # no point if we're on the last iteration
    #                 fantasy_model = botorch.models.model.ModelList(*[task.regressor for task in self.tasks])
    #                 fantasy_posterior = fantasy_model.posterior(
    #                     torch.tensor(self.normalize_inputs(new_input)).double()
    #                 )

    #                 fantasy_targets = fantasy_posterior.mean - disappointment * fantasy_posterior.variance.sqrt()
    #                 new_targets = self.unnormalize_targets(fantasy_targets.detach().numpy())

    #                 self.tell(new_table=new_table)

    #         self.forget(index=self.table.index[-n:])  # forget what you saw here

    #         return np.atleast_2d(inputs_to_sample)

    def ask(
        self,
        tasks=None,
        classifier=None,
        strategy="ei",
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

            routing_index, _ = utils.get_routing(self.read_active_dofs, unrouted_points)
            return unrouted_points[routing_index]

        if strategy.lower() == "quasi-random":
            return self.unnormalize_active_inputs(self.sampler(n=n))

        if not self._initialized:
            raise RuntimeError("The agent is not initialized!")

        self.acqf = None

        if strategy.lower() == "ei":  # maximize the expected improvement
            self.acqf = LogExpectedImprovement(
                self.task_model, best_f=self.best_sum_of_tasks, posterior_transform=self.task_scalarization
            )

        if strategy.lower() == "pi":  # maximize the expected improvement
            self.acqf = LogProbabilityOfImprovement(
                self.task_model, best_f=self.best_sum_of_tasks, posterior_transform=self.task_scalarization
            )

        if self.acqf is None:
            raise ValueError(f'Unrecognized strategy "{strategy}".')

        BATCH_SIZE = 1
        NUM_RESTARTS = 7
        RAW_SAMPLES = 512

        candidates, _ = botorch.optim.optimize_acqf(
            acq_function=self.acqf,
            bounds=torch.tensor(self.active_dof_bounds).T,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates.detach().numpy()

    def acquire(self, dof_inputs):
        """
        Acquire and digest according to the agent's acquisition and digestion plans.

        This should yield a table of sampled tasks with the same length as the sampled inputs.
        """
        try:
            uid = yield from self.acquisition_plan(
                self.dofs, dof_inputs, [*self.dets, *self.dofs, *self.passive_dofs]
            )
            products = self.digestion(self.db, uid)
            if "rejected" not in products.keys():
                products["rejected"] = False

            # compute the fitness for each task
            for index, entry in products.iterrows():
                for task in self.tasks:
                    products.loc[index, task.name] = task.get_fitness(entry)

        except Exception as err:
            raise err

        if not len(dof_inputs) == len(products):
            raise ValueError("The resulting table must be the same length as the sampled inputs!")

        return products

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

            new_table = yield from self.acquire(inputs_to_sample)

            self.tell(new_table=new_table, reuse_hypers=reuse_hypers)

    def plot_tasks(self, **kwargs):
        if self.n_dof == 1:
            self._plot_tasks_one_dof(**kwargs)

        else:
            self._plot_tasks_many_dofs(**kwargs)

    def plot_feasibility(self, **kwargs):
        if self.n_dof == 1:
            self._plot_feasibility_one_dof(**kwargs)

        else:
            self._plot_feasibility_many_dofs(**kwargs)

    def _plot_feasibility_one_dof(self, size=32):
        self.class_fig, self.class_ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, constrained_layout=True)

        self.class_ax.scatter(self.inputs.values, self.all_targets_valid.astype(int), s=size)

        x = torch.tensor(self.test_inputs_grid.reshape(-1, self.n_dof)).double()
        log_prob = self.dirichlet_classifier.log_prob(x).detach().numpy().reshape(self.test_inputs_grid.shape[:-1])

        self.class_ax.plot(self.test_inputs_grid.ravel(), np.exp(log_prob))

        self.class_ax.set_xlim(*self.active_dof_bounds[0])

    def _plot_feasibility_many_dofs(
        self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=32, gridded=None
    ):
        if gridded is None:
            gridded = self.n_dof == 2

        self.class_fig, self.class_axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True
        )

        for ax in self.class_axes.ravel():
            ax.set_xlabel(self.dofs[axes[0]].name)
            ax.set_ylabel(self.dofs[axes[1]].name)

        data_ax = self.class_axes[0].scatter(
            *self.inputs.values.T[:2], s=size, c=self.all_targets_valid.astype(int), vmin=0, vmax=1, cmap=cmap
        )

        if gridded:
            x = torch.tensor(self.test_inputs_grid.reshape(-1, self.n_dof)).double()
            log_prob = (
                self.dirichlet_classifier.log_prob(x).detach().numpy().reshape(self.test_inputs_grid.shape[:-1])
            )

            self.class_axes[1].pcolormesh(
                *np.swapaxes(self.test_inputs_grid, 0, -1),
                np.exp(log_prob).T,
                shading=shading,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

        else:
            x = torch.tensor(self.normalize_inputs(self.test_inputs)).double()
            log_prob = self.dirichlet_classifier.log_prob(x).detach().numpy()

            self.class_axes[1].scatter(
                *self.test_inputs.T[axes], s=size, c=np.exp(log_prob), vmin=0, vmax=1, cmap=cmap
            )

        self.class_fig.colorbar(data_ax, ax=self.class_axes[:2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.class_axes.ravel():
            ax.set_xlim(*self.active_dof_bounds[axes[0]])
            ax.set_ylim(*self.active_dof_bounds[axes[1]])

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
            color = DEFAULT_COLOR_LIST[itask]

            self.task_axes[itask].set_ylabel(task.name)

            task_posterior = task.regressor.posterior(
                torch.tensor(self.normalize_inputs(self.test_inputs_grid)).double()
            )
            task_mean = task_posterior.mean.detach().numpy() * task.targets_scale + task.targets_mean
            task_sigma = task_posterior.variance.sqrt().detach().numpy() * task.targets_scale

            self.task_axes[itask].scatter(self.inputs.values, task.targets, s=size, color=color)
            self.task_axes[itask].plot(self.test_active_inputs_grid.ravel(), task_mean, lw=lw, color=color)

            for z in [1, 2]:
                self.task_axes[itask].fill_between(
                    self.test_inputs_grid.ravel(),
                    (task_mean - z * task_sigma).ravel(),
                    (task_mean + z * task_sigma).ravel(),
                    lw=lw,
                    color=color,
                    alpha=0.5**z,
                )

            self.task_axes[itask].set_xlim(*self.active_dof_bounds[0])

    def _plot_tasks_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=32):
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
                *self.inputs.values.T[axes], s=size, c=task.targets, norm=task_norm, cmap=cmap
            )

            x = (
                torch.tensor(self.test_inputs_grid).double()
                if gridded
                else torch.tensor(self.test_inputs).double()
            )

            task_posterior = task.regressor.posterior(x)
            task_mean = task_posterior.mean.detach().numpy()  # * task.targets_scale + task.targets_mean
            task_sigma = task_posterior.variance.sqrt().detach().numpy()  # * task.targets_scale

            if gridded:
                self.task_axes[itask, 1].pcolormesh(
                    *np.swapaxes(self.test_inputs_grid, 0, -1),
                    task_mean.reshape(self.test_active_inputs_grid.shape[:-1]).T,
                    shading=shading,
                    cmap=cmap,
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[itask, 2].pcolormesh(
                    *np.swapaxes(self.test_inputs_grid, 0, -1),
                    task_sigma.reshape(self.normalize_inputs(self.test_inputs_grid).shape[:-1]).T,
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
            ax.set_xlim(*self.active_dof_bounds[axes[0]])
            ax.set_ylim(*self.active_dof_bounds[axes[1]])

    def normalize_active_inputs(self, inputs):
        return (inputs - self.active_dof_bounds.min(axis=1)) / self.active_dof_bounds.ptp(axis=1)

    def unnormalize_active_inputs(self, X):
        return X * self.active_dof_bounds.ptp(axis=1) + self.active_dof_bounds.min(axis=1)

    def normalize_inputs(self, inputs):
        return (inputs - self.input_bounds.min(axis=1)) / self.input_bounds.ptp(axis=1)

    def unnormalize_inputs(self, X):
        return X * self.input_bounds.ptp(axis=1) + self.input_bounds.min(axis=1)

    def normalize_targets(self, targets):
        return (targets - self.targets_mean) / (1e-20 + self.targets_scale)

    def unnormalize_targets(self, targets):
        return targets * self.targets_scale + self.targets_mean

    @property
    def test_inputs(self):
        test_passive_inputs = (
            self.passive_inputs.values[-1][None] * np.ones(len(self.test_active_inputs))[..., None]
        )
        return np.concatenate([self.test_active_inputs, test_passive_inputs], axis=-1)

    @property
    def test_inputs_grid(self):
        test_passive_inputs_grid = self.passive_inputs.values[-1] * np.ones(
            (*self.test_active_inputs_grid.shape[:-1], self.n_passive_dof)
        )
        return np.concatenate([self.test_active_inputs_grid, test_passive_inputs_grid], axis=-1)

    @property
    def inputs(self):
        return self.table.loc[:, self.dof_names].astype(float)

    @property
    def active_inputs(self):
        return self.inputs.loc[:, self.active_dof_names]

    @property
    def passive_inputs(self):
        return self.inputs.loc[:, self.passive_dof_names]

    @property
    def targets(self):
        return self.table.loc[:, self.task_names].astype(float)

    @property
    def feasible(self):
        with pd.option_context("mode.use_inf_as_null", True):
            feasible = ~self.targets.isna()
        return feasible

    @property
    def input_bounds(self):
        lower_bound = np.r_[
            self.active_dof_bounds[:, 0], np.nanmin(self.passive_inputs.astype(float).values, axis=0)
        ]
        upper_bound = np.r_[
            self.active_dof_bounds[:, 1], np.nanmax(self.passive_inputs.astype(float).values, axis=0)
        ]
        return np.c_[lower_bound, upper_bound]

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
    def read_active_dofs(self):
        return np.array([dof.read()[dof.name]["value"] for dof in self.active_dofs])

    @property
    def read_passive_dofs(self):
        return np.array([dof.read()[dof.name]["value"] for dof in self.passive_dofs])

    @property
    def read_dofs(self):
        return np.r_[self.read_active_dofs, self.read_passive_dofs]

    @property
    def active_dof_names(self):
        return [dof.name for dof in self.active_dofs]

    @property
    def passive_dof_names(self):
        return [dof.name for dof in self.passive_dofs]

    @property
    def dof_names(self):
        return [dof.name for dof in self.dofs]

    @property
    def det_names(self):
        return [det.name for det in self.dets]

    @property
    def target_names(self):
        return [task.name for task in self.tasks]

    @property
    def task_names(self):
        return [task.name for task in self.tasks]

    @property
    def task_weights(self):
        return np.array([task.weight for task in self.tasks])

    @property
    def best_sum_of_tasks(self):
        return self.targets.fillna(-np.inf).sum(axis=1).max()

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
