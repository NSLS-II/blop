import logging
import time as ttime
import warnings
from collections import OrderedDict

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
from matplotlib.patches import Patch

from .. import utils
from . import acquisition, models

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
DEFAULT_COLORMAP = "turbo"


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
# tasks: these are quantities that our self will try to optimize over

MAX_TEST_INPUTS = 2**11

AVAILABLE_ACQFS = {
    "expected_mean": {
        "identifiers": ["em", "expected_mean"],
    },
    "expected_improvement": {
        "identifiers": ["ei", "expected_improvement"],
    },
    "probability_of_improvement": {
        "identifiers": ["pi", "probability_of_improvement"],
    },
    "upper_confidence_bound": {
        "identifiers": ["ucb", "upper_confidence_bound"],
        "default_args": {"beta": 4},
    },
}


class Agent:
    def __init__(
        self,
        active_dofs,
        active_dof_bounds,
        tasks,
        db,
        **kwargs,
    ):
        """
        A Bayesian optimization self.

        Parameters
        ----------
        dofs : iterable of ophyd objects
            The degrees of freedom that the self can control, which determine the output of the model.
        bounds : iterable of lower and upper bounds
            The bounds on each degree of freedom. This should be an array of shape (n_dofs, 2).
        tasks : iterable of tasks
            The tasks which the self will try to optimize.
        acquisition : Bluesky plan generator that takes arguments (dofs, inputs, dets)
            A plan that samples the beamline for some given inputs.
        digestion : function that takes arguments (db, uid)
            A function to digest the output of the acquisition.
        db : A databroker instance.
        """

        self.active_dofs = list(np.atleast_1d(active_dofs))
        self.passive_dofs = list(np.atleast_1d(kwargs.get("passive_dofs", [])))

        for dof in self.dofs:
            dof.kind = "hinted"

        self.active_dof_bounds = np.atleast_2d(active_dof_bounds).astype(float)
        self.tasks = np.atleast_1d(tasks)
        self.db = db

        self.verbose = kwargs.get("verbose", False)
        self.ignore_acquisition_errors = kwargs.get("ignore_acquisition_errors", False)

        self.initialization = kwargs.get("initialization", None)
        self.acquisition_plan = kwargs.get("acquisition_plan", default_acquisition_plan)
        self.digestion = kwargs.get("digestion", default_digestion_plan)

        self.decoherence = kwargs.get("decoherence", False)

        self.tolerate_acquisition_errors = kwargs.get("tolerate_acquisition_errors", True)

        self.acquisition = acquisition.Acquisition()

        self.dets = np.atleast_1d(kwargs.get("detectors", []))

        for i, task in enumerate(self.tasks):
            task.index = i

        self.n_tasks = len(self.tasks)

        self.training_iter = kwargs.get("training_iter", 256)

        # make some test points for sampling

        self.normalized_test_active_inputs = utils.normalized_sobol_sampler(
            n=MAX_TEST_INPUTS, d=self.n_active_dofs
        )

        n_per_active_dim = int(np.power(MAX_TEST_INPUTS, 1 / self.n_active_dofs))

        self.normalized_test_active_inputs_grid = np.swapaxes(
            np.r_[np.meshgrid(*self.n_active_dofs * [np.linspace(0, 1, n_per_active_dim)])], 0, -1
        )

        self.table = pd.DataFrame()

        self._initialized = False
        self._train_models = True

        self.a_priori_hypers = None

    def normalize_active_inputs(self, x):
        return (x - self.active_dof_bounds.min(axis=1)) / self.active_dof_bounds.ptp(axis=1)

    def unnormalize_active_inputs(self, x):
        return x * self.active_dof_bounds.ptp(axis=1) + self.active_dof_bounds.min(axis=1)

    def active_inputs_sampler(self, n=MAX_TEST_INPUTS):
        """
        Returns $n$ quasi-randomly sampled inputs in the bounded parameter space
        """
        return self.unnormalize_active_inputs(utils.normalized_sobol_sampler(n, self.n_active_dofs))

    @property
    def dofs(self):
        return np.append(self.active_dofs, self.passive_dofs)

    @property
    def n_active_dofs(self):
        return len(self.active_dofs)

    @property
    def n_passive_dofs(self):
        return len(self.passive_dofs)

    @property
    def n_dofs(self):
        return self.n_active_dofs + self.n_passive_dofs

    @property
    def test_active_inputs(self):
        """
        A static, quasi-randomly sampled set of test active inputs.
        """
        return self.unnormalize_active_inputs(self.normalized_test_active_inputs)

    @property
    def test_active_inputs_grid(self):
        """
        A static, gridded set of test active inputs.
        """
        return self.unnormalize_active_inputs(self.normalized_test_active_inputs_grid)

    # @property
    # def input_transform(self):
    #     return botorch.models.transforms.input.Normalize(d=self.n_dofs)

    @property
    def input_transform(self):
        coefficient = torch.tensor(self.dof_bounds.ptp(axis=1)).unsqueeze(0)
        offset = torch.tensor(self.dof_bounds.min(axis=1)).unsqueeze(0)
        return botorch.models.transforms.input.AffineInputTransform(
            d=self.n_dofs, coefficient=coefficient, offset=offset
        )

    def save_data(self, filepath="./self_data.h5"):
        """
        Save the sampled inputs and targets of the self to a file, which can be used
        to initialize a future self.
        """

        self.table.to_hdf(filepath, key="table")

    def forget(self, index):
        self.tell(new_table=self.table.drop(index=index), append=False)

    def sampler(self, n):
        """
        Returns $n$ quasi-randomly sampled points on the [0,1] ^ n_active_dof hypercube using Sobol sampling.
        """
        min_power_of_two = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        subset = np.random.choice(min_power_of_two, size=n, replace=False)
        return sp.stats.qmc.Sobol(d=self.n_active_dofs, scramble=True).random(n=min_power_of_two)[subset]

    def _set_hypers(self, hypers):
        for task in self.tasks:
            task.regressor.load_state_dict(hypers[task.name])
        self.classifier.load_state_dict(hypers["classifier"])

    @property
    def hypers(self):
        hypers = {"classifier": {}}
        for key, value in self.classifier.state_dict().items():
            hypers["classifier"][key] = value
        for task in self.tasks:
            hypers[task.name] = {}
            for key, value in task.regressor.state_dict().items():
                hypers[task.name][key] = value

        return hypers

    def save_hypers(self, filepath):
        hypers = self.hypers
        with h5py.File(filepath, "w") as f:
            for model_key in hypers.keys():
                f.create_group(model_key)
                for param_key, param_value in hypers[model_key].items():
                    f[model_key].create_dataset(param_key, data=param_value)

    @staticmethod
    def load_hypers(filepath):
        hypers = {}
        with h5py.File(filepath, "r") as f:
            for model_key in f.keys():
                hypers[model_key] = OrderedDict()
                for param_key, param_value in f[model_key].items():
                    hypers[model_key][param_key] = torch.tensor(np.atleast_1d(param_value[()]))
        return hypers

    def initialize(
        self,
        acqf=None,
        n_init=4,
        data=None,
        hypers=None,
    ):
        """
        An initialization plan for the self.
        This must be run before the self can learn.
        It should be passed to a Bluesky RunEngine.
        """

        # experiment-specific stuff
        if self.initialization is not None:
            yield from self.initialization()

        if hypers is not None:
            self.a_priori_hypers = self.load_hypers(hypers)

        if data is not None:
            if type(data) == str:
                self.tell(new_table=pd.read_hdf(data, key="table"))
            else:
                self.tell(new_table=data)

        # now let's get bayesian
        elif acqf in ["qr"]:
            yield from self.learn("qr", n_iter=1, n_per_iter=n_init, route=True)

        else:
            raise Exception(
                """Could not initialize model! Either load a table, or specify an acqf from:
['qr']."""
            )

        self._initialized = True

    def tell(self, new_table=None, append=True, train=True, **kwargs):
        """
        Inform the self about new inputs and targets for the model.
        """

        new_table = pd.DataFrame() if new_table is None else new_table

        self.table = pd.concat([self.table, new_table]) if append else new_table

        self.table.loc[:, "total_fitness"] = self.table.loc[:, self.task_names].fillna(-np.inf).sum(axis=1)
        self.table.index = np.arange(len(self.table))

        skew_dims = [tuple(np.arange(self.n_active_dofs))]

        if not train:
            hypers = self.hypers

        for task in self.tasks:
            task.targets = self.targets.loc[:, task.name]

            task.feasibility = self.feasible_for_all_tasks

            if not task.feasibility.sum() >= 2:
                raise ValueError("There must be at least two feasible data points per task!")

            train_inputs = torch.tensor(self.inputs.loc[task.feasibility].values).double().unsqueeze(0)
            train_targets = (
                torch.tensor(task.targets.loc[task.feasibility].values).double().unsqueeze(0).unsqueeze(-1)
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

            task.regressor = models.LatentGP(
                train_inputs=train_inputs,
                train_targets=train_targets,
                likelihood=likelihood,
                skew_dims=skew_dims,
                input_transform=self.input_transform,
                outcome_transform=botorch.models.transforms.outcome.Standardize(m=1, batch_shape=torch.Size((1,))),
            ).double()

            task.regressor_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                task.regressor.likelihood, task.regressor
            )

        log_feas_prob_weight = np.sqrt(
            np.sum(np.nanvar(self.targets.values, axis=0) * np.square(self.task_weights))
        )

        self.task_scalarization = botorch.acquisition.objective.ScalarizedPosteriorTransform(
            weights=torch.tensor([*[task.weight for task in self.tasks], log_feas_prob_weight]).double(),
            offset=0,
        )

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.as_tensor(self.feasible_for_all_tasks.values).long(), learn_additional_noise=True
        ).double()

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=torch.tensor(self.inputs.values).double(),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            skew_dims=skew_dims,
            likelihood=dirichlet_likelihood,
            input_transform=self.input_transform,
        ).double()

        self.classifier_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.classifier.likelihood, self.classifier)

        self.feas_model = botorch.models.deterministic.GenericDeterministicModel(
            f=lambda X: -self.classifier.log_prob(X).square()
        )

        if self.a_priori_hypers is not None:
            self._set_hypers(self.a_priori_hypers)
        elif not train:
            self._set_hypers(hypers)
        else:
            self.train_models()

        self.task_model = botorch.models.model.ModelList(*[task.regressor for task in self.tasks], self.feas_model)

    def train_models(self, **kwargs):
        t0 = ttime.monotonic()
        for task in self.tasks:
            botorch.fit.fit_gpytorch_mll(task.regressor_mll, **kwargs)
        botorch.fit.fit_gpytorch_mll(self.classifier_mll, **kwargs)
        if self.verbose:
            print(f"trained models in {ttime.monotonic() - t0:.02f} seconds")

    def get_acquisition_function(self, acqf_identifier="ei", return_metadata=False, acqf_args={}, **kwargs):
        if not self._initialized:
            raise RuntimeError(
                f'Can\'t construct acquisition function "{acqf_identifier}" (the self is not initialized!)'
            )

        if acqf_identifier.lower() in AVAILABLE_ACQFS["expected_improvement"]["identifiers"]:
            acqf = botorch.acquisition.analytic.LogExpectedImprovement(
                self.task_model,
                best_f=self.best_sum_of_tasks,
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "expected improvement", "args": {}}

        elif acqf_identifier.lower() in AVAILABLE_ACQFS["probability_of_improvement"]["identifiers"]:
            acqf = botorch.acquisition.analytic.LogProbabilityOfImprovement(
                self.task_model,
                best_f=self.best_sum_of_tasks,
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "probability of improvement", "args": {}}

        elif acqf_identifier.lower() in AVAILABLE_ACQFS["expected_mean"]["identifiers"]:
            acqf = botorch.acquisition.analytic.UpperConfidenceBound(
                self.task_model,
                beta=0,
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "expected mean"}

        elif acqf_identifier.lower() in AVAILABLE_ACQFS["upper_confidence_bound"]["identifiers"]:
            beta = AVAILABLE_ACQFS["upper_confidence_bound"]["default_args"]["beta"]
            acqf = botorch.acquisition.analytic.UpperConfidenceBound(
                self.task_model,
                beta=beta,
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "upper confidence bound", "args": {"beta": beta}}

        else:
            raise ValueError(f'Unrecognized acquisition acqf_identifier "{acqf_identifier}".')

        return (acqf, acqf_meta) if return_metadata else acqf

    def ask(self, acqf_identifier="ei", n=1, route=True, return_metadata=False):
        if acqf_identifier.lower() == "qr":
            x = self.active_inputs_sampler(n=n)
            acqf_meta = {"name": "quasi-random", "args": {}}

        elif n == 1:
            x, acqf_meta = self.ask_single(acqf_identifier, return_metadata=True)
            return (x, acqf_meta) if return_metadata else x

        elif n > 1:
            for i in range(n):
                x, acqf_meta = self.ask_single(acqf_identifier, return_metadata=True)

                if i < (n - 1):
                    task_samples = [
                        task.regressor.posterior(torch.tensor(x)).sample().item() for task in self.tasks
                    ]
                    fantasy_table = pd.DataFrame(
                        np.append(x, task_samples)[None], columns=[*self.dof_names, *self.task_names]
                    )
                    self.tell(fantasy_table, train=False)

            x = self.active_inputs.iloc[-n:].values

            if n > 1:
                self.forget(self.table.index[-(n - 1) :])

        if route:
            x = x[utils.route(self.read_active_dofs, x)]

        return (x, acqf_meta) if return_metadata else x

    def ask_single(
        self,
        acqf_identifier="ei",
        return_metadata=False,
    ):
        """
        The next $n$ points to sample, recommended by the self.
        """

        t0 = ttime.monotonic()

        acqf, acqf_meta = self.get_acquisition_function(acqf_identifier=acqf_identifier, return_metadata=True)

        BATCH_SIZE = 1
        NUM_RESTARTS = 8
        RAW_SAMPLES = 256

        candidates, _ = botorch.optim.optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor(self.dof_bounds).T,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        )

        x = candidates.detach().numpy()[..., self.dof_is_active_mask]

        if self.verbose:
            print(f"found point {x} in {ttime.monotonic() - t0:.02f} seconds")

        return (x, acqf_meta) if return_metadata else x

    def acquire(self, active_inputs):
        """
        Acquire and digest according to the self's acquisition and digestion plans.

        This should yield a table of sampled tasks with the same length as the sampled inputs.
        """
        try:
            uid = yield from self.acquisition_plan(
                self.dofs, active_inputs, [*self.dets, *self.dofs, *self.passive_dofs]
            )

            products = self.digestion(self.db, uid)

            # compute the fitness for each task
            for index, entry in products.iterrows():
                for task in self.tasks:
                    products.loc[index, task.name] = task.get_fitness(entry)

        except Exception as error:
            if not self.tolerate_acquisition_errors:
                raise error
            logging.warning(f"Error in acquisition/digestion: {repr(error)}")
            products = pd.DataFrame(active_inputs, columns=self.active_dof_names)
            for task in self.tasks:
                products.loc[:, task.name] = np.nan

        if not len(active_inputs) == len(products):
            raise ValueError("The table returned by the digestion must be the same length as the sampled inputs!")

        return products

    def learn(
        self,
        acqf_identifier,
        n_iter=1,
        n_per_iter=1,
        reuse_hypers=True,
        upsample=1,
        verbose=True,
        plots=[],
        **kwargs,
    ):
        """
        This iterates the learning algorithm, looping over ask -> acquire -> tell.
        It should be passed to a Bluesky RunEngine.
        """

        for iteration in range(n_iter):
            x, acqf_meta = self.ask(n=n_per_iter, acqf_identifier=acqf_identifier, return_metadata=True, **kwargs)

            new_table = yield from self.acquire(x)

            new_table.loc[:, "acqf"] = acqf_meta["name"]

            self.tell(new_table=new_table, reuse_hypers=reuse_hypers)

    def normalize_inputs(self, inputs):
        return (inputs - self.input_bounds.min(axis=1)) / self.input_bounds.ptp(axis=1)

    def unnormalize_inputs(self, X):
        return X * self.input_bounds.ptp(axis=1) + self.input_bounds.min(axis=1)

    def normalize_targets(self, targets):
        return (targets - self.targets_mean) / (1e-20 + self.targets_scale)

    def unnormalize_targets(self, targets):
        return targets * self.targets_scale + self.targets_mean

    @property
    def batch_dimension(self):
        return self.dof_names.index("training_batch") if "training_batch" in self.dof_names else None

    @property
    def test_inputs(self):
        test_passive_inputs = self.read_passive_dofs[None] * np.ones(len(self.test_active_inputs))[..., None]
        return np.concatenate([self.test_active_inputs, test_passive_inputs], axis=-1)

    @property
    def test_inputs_grid(self):
        test_passive_inputs_grid = self.read_passive_dofs * np.ones(
            (*self.test_active_inputs_grid.shape[:-1], self.n_passive_dofs)
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

    # @property
    # def feasible(self):
    #     with pd.option_context("mode.use_inf_as_null", True):
    #         feasible = ~self.targets.isna()
    #     return feasible

    @property
    def feasible_for_all_tasks(self):
        with pd.option_context("mode.use_inf_as_null", True):
            feasible = ~self.targets.isna().any(axis=1)
            for task in self.tasks:
                if task.min is not None:
                    feasible &= self.targets.loc[:, task.name].values > task.transform(task.min)
        return feasible

    # @property
    # def input_bounds(self):
    #     lower_bound = np.r_[
    #         self.active_dof_bounds[:, 0], np.nanmin(self.passive_inputs.astype(float).values, axis=0)
    #     ]
    #     upper_bound = np.r_[
    #         self.active_dof_bounds[:, 1], np.nanmax(self.passive_inputs.astype(float).values, axis=0)
    #     ]
    #     return np.c_[lower_bound, upper_bound]

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
    def latest_passive_dof_values(self):
        passive_inputs = self.passive_inputs
        return [passive_inputs.loc[passive_inputs.last_valid_index(), col] for col in passive_inputs.columns]

    @property
    def passive_dof_bounds(self):
        # food for thought: should this be the current values, or the latest recorded values?
        # the former leads to weird extrapolation (especially for time), and the latter to some latency.
        # let's go with the second way for now
        return np.outer(self.read_passive_dofs, [1.0, 1.0])

    @property
    def dof_is_active_mask(self):
        return np.r_[np.ones(self.n_active_dofs), np.zeros(self.n_passive_dofs)].astype(bool)

    @property
    def dof_bounds(self):
        return np.r_[self.active_dof_bounds, self.passive_dof_bounds]

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

    def plot_tasks(self, **kwargs):
        if self.n_active_dofs == 1:
            self._plot_tasks_one_dof(**kwargs)

        else:
            self._plot_tasks_many_dofs(**kwargs)

    def plot_feasibility(self, **kwargs):
        if self.n_active_dofs == 1:
            self._plot_feas_one_dof(**kwargs)

        else:
            self._plot_feas_many_dofs(**kwargs)

    def plot_acquisition(self, **kwargs):
        if self.n_active_dofs == 1:
            self._plot_acq_one_dof(**kwargs)

        else:
            self._plot_acq_many_dofs(**kwargs)

    def _plot_feas_one_dof(self, size=32):
        self.class_fig, self.class_ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, constrained_layout=True)

        self.class_ax.scatter(self.inputs.values, self.feasible_for_all_tasks.astype(int), s=size)

        x = torch.tensor(self.test_inputs_grid.reshape(-1, self.n_dofs)).double()
        log_prob = self.classifier.log_prob(x).detach().numpy().reshape(self.test_inputs_grid.shape[:-1])

        self.class_ax.plot(self.test_inputs_grid.ravel(), np.exp(log_prob))

        self.class_ax.set_xlim(*self.active_dof_bounds[0])

    def _plot_feas_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=32, gridded=None):
        if gridded is None:
            gridded = self.n_dofs == 2

        self.class_fig, self.class_axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True
        )

        for ax in self.class_axes.ravel():
            ax.set_xlabel(self.dofs[axes[0]].name)
            ax.set_ylabel(self.dofs[axes[1]].name)

        data_ax = self.class_axes[0].scatter(
            *self.inputs.values.T[:2], s=size, c=self.feasible_for_all_tasks.astype(int), vmin=0, vmax=1, cmap=cmap
        )

        if gridded:
            x = torch.tensor(self.test_inputs_grid.reshape(-1, self.n_dofs)).double()
            log_prob = self.classifier.log_prob(x).detach().numpy().reshape(self.test_inputs_grid.shape[:-1])

            self.class_axes[1].pcolormesh(
                *np.swapaxes(self.test_inputs_grid, 0, -1),
                np.exp(log_prob).T,
                shading=shading,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

        else:
            x = torch.tensor(self.test_inputs).double()
            log_prob = self.classifier.log_prob(x).detach().numpy()

            self.class_axes[1].scatter(
                *x.detach().numpy().T[axes], s=size, c=np.exp(log_prob), vmin=0, vmax=1, cmap=cmap
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

            task_posterior = task.regressor.posterior(torch.tensor(self.test_inputs_grid).double())
            task_mean = task_posterior.mean.detach().numpy().ravel()
            task_sigma = task_posterior.variance.sqrt().detach().numpy().ravel()

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
            gridded = self.n_dofs == 2

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
                    task_sigma.reshape(self.test_inputs_grid.shape[:-1]).T,
                    shading=shading,
                    cmap=cmap,
                )

            else:
                self.task_axes[itask, 1].scatter(
                    *x.detach().numpy().T[axes], s=size, c=task_mean, norm=task_norm, cmap=cmap
                )
                sigma_ax = self.task_axes[itask, 2].scatter(
                    *x.detach().numpy().T[axes], s=size, c=task_sigma, cmap=cmap
                )

            self.task_fig.colorbar(data_ax, ax=self.task_axes[itask, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(sigma_ax, ax=self.task_axes[itask, 2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self.active_dof_bounds[axes[0]])
            ax.set_ylim(*self.active_dof_bounds[axes[1]])

    def _plot_acq_one_dof(self, size=32, lw=1e0, **kwargs):
        acqf_names = np.atleast_1d(kwargs.get("acqf", "ei"))

        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acqf_names),
            figsize=(6 * len(acqf_names), 6),
            sharex=True,
            constrained_layout=True,
        )

        self.acq_axes = np.atleast_1d(self.acq_axes)

        for iacqf, acqf_name in enumerate(acqf_names):
            color = DEFAULT_COLOR_LIST[0]

            acqf, acqf_meta = self.get_acquisition_function(acqf_name, return_metadata=True)

            *grid_shape, dim = self.test_inputs_grid.shape
            x = torch.tensor(self.test_inputs_grid.reshape(-1, 1, dim)).double()
            obj = acqf.forward(x)

            if acqf_name in ["ei", "pi"]:
                obj = obj.exp()

            self.acq_axes[iacqf].set_title(acqf_meta["name"])
            self.acq_axes[iacqf].plot(
                self.test_active_inputs_grid.ravel(), obj.detach().numpy().ravel(), lw=lw, color=color
            )

            self.acq_axes[iacqf].set_xlim(*self.active_dof_bounds[0])

    def _plot_acq_many_dofs(
        self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=32, **kwargs
    ):
        acqf_names = np.atleast_1d(kwargs.get("acqf", "ei"))

        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acqf_names),
            figsize=(4 * len(acqf_names), 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        if gridded is None:
            gridded = self.n_active_dofs == 2

        self.acq_axes = np.atleast_1d(self.acq_axes)
        self.acq_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        for iacqf, acqf_name in enumerate(acqf_names):
            acqf, acqf_meta = self.get_acquisition_function(acqf_name, return_metadata=True)

            if gridded:
                *grid_shape, dim = self.test_inputs_grid.shape
                x = torch.tensor(self.test_inputs_grid.reshape(-1, 1, dim)).double()
                obj = acqf.forward(x)

                if acqf_name in ["ei", "pi"]:
                    obj = obj.exp()

                self.acq_axes[iacqf].set_title(acqf_meta["name"])
                obj_ax = self.acq_axes[iacqf].pcolormesh(
                    *np.swapaxes(self.test_inputs_grid, 0, -1)[axes],
                    obj.detach().numpy().reshape(grid_shape).T,
                    shading=shading,
                    cmap=cmap,
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

            else:
                *inputs_shape, dim = self.test_inputs.shape
                x = torch.tensor(self.test_inputs.reshape(-1, 1, dim)).double()
                obj = acqf.forward(x)

                if acqf_name in ["ei", "pi"]:
                    obj = obj.exp()

                self.acq_axes[iacqf].set_title(acqf_meta["name"])
                obj_ax = self.acq_axes[iacqf].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=obj.detach().numpy().reshape(inputs_shape),
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

        for ax in self.acq_axes.ravel():
            ax.set_xlim(*self.active_dof_bounds[axes[0]])
            ax.set_ylim(*self.active_dof_bounds[axes[1]])

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

    def plot_history(self, x_key="index", show_all_tasks=False):
        x = getattr(self.table, x_key).values

        num_task_plots = 1
        if show_all_tasks:
            num_task_plots = self.n_tasks + 1

        self.n_tasks + 1 if self.n_tasks > 1 else 1

        hist_fig, hist_axes = plt.subplots(
            num_task_plots, 1, figsize=(6, 4 * num_task_plots), sharex=True, constrained_layout=True, dpi=200
        )
        hist_axes = np.atleast_1d(hist_axes)

        unique_strategies, acqf_index, acqf_inverse = np.unique(
            self.table.acqf, return_index=True, return_inverse=True
        )

        sample_colors = np.array(DEFAULT_COLOR_LIST)[acqf_inverse]

        if show_all_tasks:
            for itask, task in enumerate(self.tasks):
                y = task.targets.values
                hist_axes[itask].scatter(x, y, c=sample_colors)
                hist_axes[itask].plot(x, y, lw=5e-1, c="k")
                hist_axes[itask].set_ylabel(task.name)

        y = self.table.total_fitness

        cummax_y = np.array([np.nanmax(y[: i + 1]) for i in range(len(y))])

        hist_axes[-1].scatter(x, y, c=sample_colors)
        hist_axes[-1].plot(x, y, lw=5e-1, c="k")

        hist_axes[-1].plot(x, cummax_y, lw=5e-1, c="k", ls=":")

        hist_axes[-1].set_ylabel("total_fitness")
        hist_axes[-1].set_xlabel(x_key)

        handles = []
        for i_acqf, acqf in enumerate(unique_strategies):
            #        i_acqf = np.argsort(acqf_index)[i_handle]
            handles.append(Patch(color=DEFAULT_COLOR_LIST[i_acqf], label=acqf))
        legend = hist_axes[0].legend(handles=handles, fontsize=8)
        legend.set_title("acquisition function")

    # plot_history(self, x_key="time")

    # plt.savefig("bo-history.pdf", dpi=256)
