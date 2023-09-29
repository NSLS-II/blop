import logging
import os
import time as ttime
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Mapping

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
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from .. import utils
from . import acquisition, models
from .acquisition import default_acquisition_plan
from .digestion import default_digestion_function

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")


MAX_TEST_INPUTS = 2**11

TASK_CONFIG = {}

TASK_TRANSFORMS = {"log": lambda x: np.log(x)}

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
DEFAULT_COLORMAP = "turbo"
DEFAULT_SCATTER_SIZE = 16

DEFAULT_MINIMUM_SNR = 2e1


def _validate_and_prepare_dofs(dofs):
    for i_dof, dof in enumerate(dofs):
        if not isinstance(dof, Mapping):
            raise ValueError("Supplied dofs must be an iterable of mappings (e.g. a dict)!")
        if "device" not in dof.keys():
            raise ValueError("Each DOF must have a device!")

        dof["device"].kind = "hinted"
        dof["name"] = dof["device"].name if hasattr(dof["device"], "name") else f"x{i_dof+1}"

        if "limits" not in dof.keys():
            dof["limits"] = (-np.inf, np.inf)
        dof["limits"] = tuple(np.array(dof["limits"], dtype=float))

        if "tags" not in dof.keys():
            dof["tags"] = []

        # dofs are passive by default
        dof["kind"] = dof.get("kind", "passive")
        if dof["kind"] not in ["active", "passive"]:
            raise ValueError('DOF kinds must be one of "active" or "passive"')

        # active dofs are on by default, passive dofs are off by default
        dof["mode"] = dof.get("mode", "on" if dof["kind"] == "active" else "off")
        if dof["mode"] not in ["on", "off"]:
            raise ValueError('DOF modes must be one of "on" or "off"')

    dof_names = [dof["device"].name for dof in dofs]

    # check that dof names are unique
    unique_dof_names, counts = np.unique(dof_names, return_counts=True)
    duplicate_dof_names = unique_dof_names[counts > 1]
    if len(duplicate_dof_names) > 0:
        raise ValueError(f'Duplicate name(s) in supplied dofs: "{duplicate_dof_names}"')

    return list(dofs)


def _validate_and_prepare_tasks(tasks):
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("Supplied tasks must be an iterable of mappings (e.g. a dict)!")
        if task["kind"] not in ["minimize", "maximize"]:
            raise ValueError('"mode" must be specified as either "minimize" or "maximize"')
        if "name" not in task.keys():
            task["name"] = task["key"]
        if "weight" not in task.keys():
            task["weight"] = 1
        if "limits" not in task.keys():
            task["limits"] = (-np.inf, np.inf)
        if "min_snr" not in task.keys():
            task["min_snr"] = DEFAULT_MINIMUM_SNR

    task_keys = [task["key"] for task in tasks]
    unique_task_keys, counts = np.unique(task_keys, return_counts=True)
    duplicate_task_keys = unique_task_keys[counts > 1]
    if len(duplicate_task_keys) > 0:
        raise ValueError(f'Duplicate key(s) in supplied tasks: "{duplicate_task_keys}"')

    return list(tasks)


class Agent:
    def __init__(
        self,
        dofs,
        tasks,
        db,
        **kwargs,
    ):
        """
        A Bayesian optimization self.

        Parameters
        ----------
        dofs : iterable of ophyd objects
            The degrees of freedom that the agent can control, which determine the output of the model.
        bounds : iterable of lower and upper bounds
            The bounds on each degree of freedom. This should be an array of shape (n_dofs, 2).
        tasks : iterable of tasks
            The tasks which the agent will try to optimize.
        acquisition : Bluesky plan generator that takes arguments (dofs, inputs, dets)
            A plan that samples the beamline for some given inputs.
        digestion : function that takes arguments (db, uid)
            A function to digest the output of the acquisition.
        db : A databroker instance.
        """

        # DOFs are parametrized by kind ("active" or "passive") and mode ("on" or "off")
        #
        # below are the behaviors of DOFs of each kind and mode:
        #
        # "read": the agent will read the input on every acquisition (all dofs are always read)
        # "move": the agent will try to set and optimize over these (there must be at least one of these)
        # "input" means that the agent will use the value to make its posterior
        #
        #
        #             active             passive
        #     +---------------------+---------------+
        #  on |  read, input, move  |  read, input  |
        #     +---------------------+---------------+
        # off |  read               |  read         |
        #     +---------------------+---------------+
        #
        #

        self.dofs = _validate_and_prepare_dofs(np.atleast_1d(dofs))
        self.tasks = _validate_and_prepare_tasks(np.atleast_1d(tasks))
        self.db = db

        self.verbose = kwargs.get("verbose", False)
        self.allow_acquisition_errors = kwargs.get("allow_acquisition_errors", True)
        self.initialization = kwargs.get("initialization", None)
        self.acquisition_plan = kwargs.get("acquisition_plan", default_acquisition_plan)
        self.digestion = kwargs.get("digestion", default_digestion_function)
        self.dets = list(np.atleast_1d(kwargs.get("dets", [])))

        self.trigger_delay = kwargs.get("trigger_delay", 0)

        self.acq_func_config = kwargs.get("acq_func_config", acquisition.config)

        self.sample_center_on_init = kwargs.get("sample_center_on_init", False)

        self.table = pd.DataFrame()

        self._initialized = False
        self._train_models = True
        self.a_priori_hypers = None

        self.plots = {"tasks": {}}

    def reset(self):
        """
        Reset the agent.
        """
        self.table = pd.DataFrame()
        self._initialized = False

    def initialize(
        self,
        acq_func=None,
        n=4,
        data=None,
        hypers=None,
    ):
        """
        An initialization plan for the self.
        This must be run before the agent can learn.
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
        elif acq_func in ["qr"]:
            if self.sample_center_on_init:
                new_table = yield from self.acquire(self.acq_func_bounds.mean(axis=0))
                new_table.loc[:, "acq_func"] = "sample_center_on_init"
                self.tell(new_table=new_table, train=False)
            yield from self.learn("qr", iterations=1, n=n, route=True)

        else:
            raise Exception(
                """Could not initialize model! Either load a table, or specify an acq_func from:
['qr']."""
            )

        self._initialized = True

    def tell(self, new_table=None, append=True, train=True, **kwargs):
        """
        Inform the agent about new inputs and targets for the model.
        """

        new_table = pd.DataFrame() if new_table is None else new_table
        self.table = pd.concat([self.table, new_table]) if append else new_table
        self.table.index = np.arange(len(self.table))

        skew_dims = self.latent_dim_tuples

        if self._initialized:
            cached_hypers = self.hypers

        inputs = self.inputs.loc[:, self._subset_dof_names(mode="on")].values

        for i, task in enumerate(self.tasks):
            self.table.loc[:, f"{task['key']}_fitness"] = targets = self._get_task_fitness(i)
            train_index = ~np.isnan(targets)

            if not train_index.sum() >= 2:
                raise ValueError("There must be at least two valid data points per task!")

            train_inputs = torch.tensor(inputs[train_index]).double()
            train_targets = torch.tensor(targets[train_index]).double().unsqueeze(-1)  # .unsqueeze(0)

            # for constructing the log normal noise prior
            target_snr = 2e2
            scale = 1e0
            loc = np.log(1 / target_snr**2) + scale**2

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(1e-4).square(),
                    torch.tensor(1 / task["min_snr"]).square(),
                ),
                #noise_prior=gpytorch.priors.torch_priors.LogNormalPrior(loc=loc, scale=scale),
            ).double()

            outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)  # , batch_shape=torch.Size((1,)))

            task["model"] = models.LatentGP(
                train_inputs=train_inputs,
                train_targets=train_targets,
                likelihood=likelihood,
                skew_dims=skew_dims,
                input_transform=self._subset_input_transform(mode="on"),
                outcome_transform=outcome_transform,
            ).double()

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            self.all_tasks_valid.long(), learn_additional_noise=True
        ).double()

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=torch.tensor(inputs).double(),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            skew_dims=skew_dims,
            likelihood=dirichlet_likelihood,
            input_transform=self._subset_input_transform(mode="on"),
        ).double()

        if self.a_priori_hypers is not None:
            self._set_hypers(self.a_priori_hypers)
        elif not train:
            self._set_hypers(cached_hypers)
        else:
            try:
                self.train_models()
            except botorch.exceptions.errors.ModelFittingError:
                if self._initialized:
                    self._set_hypers(cached_hypers)
                else:
                    raise RuntimeError("Could not fit model on initialization!")

        self.constraint = GenericDeterministicModel(f=lambda x: self.classifier.probabilities(x)[..., -1].squeeze(-1))

    @property
    def model(self):
        """
        A model encompassing all the tasks. A single GP in the single-task case, or a model list.
        """
        return ModelListGP(*[task["model"] for task in self.tasks]) if self.num_tasks > 1 else self.tasks[0]["model"]

    def _get_task_fitness(self, task_index):
        """
        Returns the fitness for a task given the task index.
        """
        task = self.tasks[task_index]

        targets = self.table.loc[:, task["key"]].values.copy()

        # check that task values are inside acceptable values
        valid = (targets > task["limits"][0]) & (targets < task["limits"][1])
        targets = np.where(valid, targets, np.nan)

        # transform if needed
        if "transform" in task.keys():
            if task["transform"] == "log":
                targets = np.where(targets > 0, np.log(targets), np.nan)

        if task["kind"] == "minimize":
            targets *= -1

        return targets

    @property
    def fitnesses(self):
        """
        Returns a (num_tasks x n_obs) array of fitnesses
        """
        return torch.cat([torch.tensor(self._get_task_fitness(i))[..., None] for i in range(self.num_tasks)], dim=1)

    @property
    def scalarized_fitness(self):
        return (self.fitnesses * self.task_weights).sum(axis=-1)

    @property
    def best_scalarized_fitness(self):
        f = self.scalarized_fitness
        return np.where(np.isnan(f), -np.inf, f).max()

    @property
    def all_tasks_valid(self):
        return ~torch.isnan(self.scalarized_fitness)

    @property
    def target_names(self):
        return [f'{task["key"]}_fitness' for task in self.tasks]

    @property
    def test_inputs_grid(self):
        n_side = int(MAX_TEST_INPUTS ** (1 / self._len_subset_dofs(kind="active", mode="on")))
        return torch.tensor(
            np.r_[
                np.meshgrid(
                    *[
                        np.linspace(*dof["limits"], n_side)
                        if dof["kind"] == "active"
                        else dof["device"].read()[dof["device"].name]["value"]
                        for dof in self._subset_dofs(mode="on")
                    ]
                )
            ]
        ).swapaxes(0, -1)

    @property
    def acq_func_bounds(self):
        return torch.tensor(
            [
                dof["limits"] if dof["kind"] == "active" else tuple(2 * [dof["device"].read()[dof["device"].name]["value"]])
                for dof in self.dofs
                if dof["mode"] == "on"
            ]
        ).T

    @property
    def num_tasks(self):
        return len(self.tasks)

    @property
    def det_names(self):
        return [det.name for det in self.dets]

    @property
    def task_keys(self):
        return [task["key"] for task in self.tasks]

    @property
    def task_models(self):
        return [task["model"] for task in self.tasks]

    @property
    def task_weights(self):
        return torch.tensor([task["weight"] for task in self.tasks], dtype=torch.float64)

    @property
    def task_signs(self):
        return torch.tensor([(1 if task["kind"] == "maximize" else -1) for task in self.tasks], dtype=torch.long)

    def _dof_kind_mask(self, kind=None):
        return [dof["kind"] == kind if kind is not None else True for dof in self.dofs]

    def _dof_mode_mask(self, mode=None):
        return [dof["mode"] == mode if mode is not None else True for dof in self.dofs]

    def _dof_tags_mask(self, tags=[]):
        return [np.isin(dof["tags"], tags).any() if tags else True for dof in self.dofs]

    def _dof_mask(self, kind=None, mode=None, tags=[]):
        return [
            (k and m and t)
            for k, m, t in zip(self._dof_kind_mask(kind), self._dof_mode_mask(mode), self._dof_tags_mask(tags))
        ]

    def activate_dofs(self, kind=None, mode=None, tags=[]):
        for dof in self._subset_dofs(kind, mode, tags):
            dof["mode"] = "on"

    def deactivate_dofs(self, kind=None, mode=None, tags=[]):
        for dof in self._subset_dofs(kind, mode, tags):
            dof["mode"] = "off"

    def _subset_dofs(self, kind=None, mode=None, tags=[]):
        return [dof for dof, m in zip(self.dofs, self._dof_mask(kind, mode, tags)) if m]

    def _len_subset_dofs(self, kind=None, mode=None, tags=[]):
        return len(self._subset_dofs(kind, mode, tags))

    def _subset_devices(self, kind=None, mode=None, tags=[]):
        return [dof["device"] for dof in self._subset_dofs(kind, mode, tags)]

    def _read_subset_devices(self, kind=None, mode=None, tags=[]):
        return [device.read()[device.name]["value"] for device in self._subset_devices(kind, mode, tags)]

    def _subset_dof_names(self, kind=None, mode=None, tags=[]):
        return [device.name for device in self._subset_devices(kind, mode, tags)]

    def _subset_dof_limits(self, kind=None, mode=None, tags=[]):
        dofs_subset = self._subset_dofs(kind, mode, tags)
        if len(dofs_subset) > 0:
            return torch.tensor([dof["limits"] for dof in dofs_subset], dtype=torch.float64).T
        return torch.empty((2, 0))

    @property
    def latent_dim_tuples(self):
        """
        Returns a list of tuples, where each tuple represent a group of dimension to find a latent representation of.
        """

        latent_dim_labels = [dof.get("latent_group", str(uuid.uuid4())) for dof in self._subset_dofs(mode="on")]
        u, uinv = np.unique(latent_dim_labels, return_inverse=True)

        return [tuple(np.where(uinv == i)[0]) for i in range(len(u))]

    def test_inputs(self, n=MAX_TEST_INPUTS):
        return utils.sobol_sampler(self.acq_func_bounds, n=n)

    def _subset_input_transform(self, kind=None, mode=None, tags=[]):
        limits = self._subset_dof_limits(kind, mode, tags)
        offset = limits.min(dim=0).values
        coefficient = limits.max(dim=0).values - offset
        return botorch.models.transforms.input.AffineInputTransform(
            d=limits.shape[-1], coefficient=coefficient, offset=offset
        )

    def _subset_inputs_sampler(self, kind=None, mode=None, tags=[], n=MAX_TEST_INPUTS):
        """
        Returns $n$ quasi-randomly sampled inputs in the bounded parameter space
        """
        transform = self._subset_input_transform(kind, mode, tags)
        return transform.untransform(utils.normalized_sobol_sampler(n, d=self._len_subset_dofs(kind, mode, tags)))

    def save_data(self, filepath="./self_data.h5"):
        """
        Save the sampled inputs and targets of the agent to a file, which can be used
        to initialize a future self.
        """

        self.table.to_hdf(filepath, key="table")

    def forget(self, index):
        self.tell(new_table=self.table.drop(index=index), append=False, train=False)

    def sampler(self, n):
        """
        Returns $n$ quasi-randomly sampled points on the [0,1] ^ n_active_dof hypercube using Sobol sampling.
        """
        min_power_of_two = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        subset = np.random.choice(min_power_of_two, size=n, replace=False)
        return sp.stats.qmc.Sobol(d=self._len_subset_dofs(kind="active", mode="on"), scramble=True).random(
            n=min_power_of_two
        )[subset]

    def _set_hypers(self, hypers):
        for task in self.tasks:
            task["model"].load_state_dict(hypers[task["key"]])
        self.classifier.load_state_dict(hypers["classifier"])

    @property
    def hypers(self):
        hypers = {"classifier": {}}
        for key, value in self.classifier.state_dict().items():
            hypers["classifier"][key] = value
        for task in self.tasks:
            hypers[task["key"]] = {}
            for key, value in task["model"].state_dict().items():
                hypers[task["key"]][key] = value

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

    def train_models(self, **kwargs):
        t0 = ttime.monotonic()
        for task in self.tasks:
            model = task["model"]
            botorch.fit.fit_gpytorch_mll(gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model), **kwargs)
        botorch.fit.fit_gpytorch_mll(
            gpytorch.mlls.ExactMarginalLogLikelihood(self.classifier.likelihood, self.classifier), **kwargs
        )
        if self.verbose:
            print(f"trained models in {ttime.monotonic() - t0:.02f} seconds")

    @property
    def acq_func_info(self):
        entries = []
        for k, d in self.acq_func_config.items():
            ret = ""
            ret += f'{d["pretty_name"].upper()} (identifiers: {d["identifiers"]})\n'
            ret += f'-> {d["description"]}'
            entries.append(ret)

        print("\n\n".join(entries))

    def ask(self, acq_func_identifier="qei", n=1, route=True, sequential=True, return_metadata=False, **acq_func_kwargs):
        """
        Ask the agent for the best point to sample, given an acquisition function.

        acq_func_identifier: which acquisition function to use
        n: how many points to get
        """

        acq_func_name = acquisition.parse_acq_func(acq_func_identifier)
        acq_func_type = acquisition.config[acq_func_name]["type"]

        start_time = ttime.monotonic()

        if acq_func_type in ["analytic", "monte_carlo"]:
            if not self._initialized:
                raise RuntimeError(
                    f'Can\'t construct acquisition function "{acq_func_identifier}" (the agent is not initialized!)'
                )

            if acq_func_type == "analytic" and n > 1:
                raise ValueError("Can't generate multiple design points for analytic acquisition functions.")

            acq_func, acq_func_meta = self.get_acquisition_function(
                acq_func_identifier=acq_func_identifier, return_metadata=True
            )

            NUM_RESTARTS = 8
            RAW_SAMPLES = 512

            candidates, _ = botorch.optim.optimize_acqf(
                acq_function=acq_func,
                bounds=self.acq_func_bounds,
                q=n,
                sequential=sequential,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            )

            x = candidates.numpy().astype(float)

            active_X = x[..., [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]]
            passive_X = x[..., [dof["kind"] != "active" for dof in self._subset_dofs(mode="on")]]
            acq_func_meta["passive_values"] = passive_X

        else:
            if acq_func_identifier.lower() == "qr":
                active_X = self._subset_inputs_sampler(n=n, kind="active", mode="on").squeeze(1).numpy()
                acq_func_meta = {"name": "quasi-random", "args": {}}

        acq_func_meta["duration"] = duration = ttime.monotonic() - start_time

        if self.verbose:
            print(f"found points {active_X} in {duration:.02f} seconds")

        if route and n > 1:
            active_X = active_X[utils.route(self._read_subset_devices(kind="active", mode="on"), active_X)]

        return (active_X, acq_func_meta) if return_metadata else active_X

    def acquire(self, active_inputs):
        """
        Acquire and digest according to the self's acquisition and digestion plans.

        This should yield a table of sampled tasks with the same length as the sampled inputs.
        """
        try:
            active_devices = self._subset_devices(kind="active", mode="on")
            passive_devices = [*self._subset_devices(kind="passive"), *self._subset_devices(kind="active", mode="off")]

            uid = yield from self.acquisition_plan(
                active_devices,
                active_inputs.astype(float),
                [*self.dets, *passive_devices],
                delay=self.trigger_delay,
            )

            products = self.digestion(self.db, uid)

            # compute the fitness for each task
            # for index, entry in products.iterrows():
            #     for task in self.tasks:
            #         products.loc[index, task["key"]] = getattr(entry, task["key"])

        except Exception as error:
            if not self.allow_acquisition_errors:
                raise error
            logging.warning(f"Error in acquisition/digestion: {repr(error)}")
            products = pd.DataFrame(active_inputs, columns=self._subset_dof_names(kind="active", mode="on"))
            for task in self.tasks:
                products.loc[:, task["key"]] = np.nan

        if not len(active_inputs) == len(products):
            raise ValueError("The table returned by the digestion must be the same length as the sampled inputs!")

        return products

    def learn(
        self,
        acq_func_identifier,
        iterations=1,
        n=1,
        reuse_hypers=True,
        train=True,
        upsample=1,
        verbose=True,
        plots=[],
        **kwargs,
    ):
        """
        This iterates the learning algorithm, looping over ask -> acquire -> tell.
        It should be passed to a Bluesky RunEngine.
        """

        for i in range(iterations):
            x, acq_func_meta = self.ask(
                n=n, acq_func_identifier=acq_func_identifier, return_metadata=True, **kwargs
            )

            new_table = yield from self.acquire(x)
            new_table.loc[:, "acq_func"] = acq_func_meta["name"]
            self.tell(new_table=new_table, train=train)

    @property
    def inputs(self):
        return self.table.loc[:, self._subset_dof_names(mode="on")].astype(float)

    @property
    def best_inputs(self):
        return self.inputs.values[np.nanargmax(self.scalarized_fitness)]

    def go_to(self, inputs):
        args = []
        for dof, value in zip(self._subset_dofs(mode="on"), np.atleast_1d(inputs).T):
            if dof["kind"] == "active":
                args.append(dof["device"])
                args.append(value)
        yield from bps.mv(*args)

    def go_to_best(self):
        yield from self.go_to(self.best_inputs)

    def plot_tasks(self, **kwargs):
        if self._len_subset_dofs(kind="active", mode="on") == 1:
            self._plot_tasks_one_dof(**kwargs)
        else:
            self._plot_tasks_many_dofs(**kwargs)

    def _plot_tasks_one_dof(self, size=16, lw=1e0):
        self.task_fig, self.task_axes = plt.subplots(
            self.num_tasks,
            1,
            figsize=(6, 4 * self.num_tasks),
            sharex=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_1d(self.task_axes)

        for task_index, task in enumerate(self.tasks):
            task_plots = self.plots["tasks"][task["name"]] = {}

            task_fitness = self._get_task_fitness(task_index=task_index)

            color = DEFAULT_COLOR_LIST[task_index]

            self.task_axes[task_index].set_ylabel(task["key"])

            x = self.test_inputs_grid
            task_posterior = task["model"].posterior(x)
            task_mean = task_posterior.mean.detach().numpy()
            task_sigma = task_posterior.variance.sqrt().detach().numpy()

            sampled_inputs = self.inputs.values[:, self._dof_mask(kind="active", mode="on")][:,0]
            task_plots["sampled"] = self.task_axes[task_index].scatter(sampled_inputs, task_fitness, s=size, color=color)

            on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]

            for z in [0, 1, 2]:
                self.task_axes[task_index].fill_between(
                    x[..., on_dofs_are_active_mask].squeeze(),
                    (task_mean - z * task_sigma).squeeze(),
                    (task_mean + z * task_sigma).squeeze(),
                    lw=lw,
                    color=color,
                    alpha=0.5**z,
                )

            self.task_axes[task_index].set_xlim(self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_tasks_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16):
        if gridded is None:
            gridded = self._len_subset_dofs(kind="active", mode="on") == 2

        self.task_fig, self.task_axes = plt.subplots(
            len(self.tasks),
            3,
            figsize=(10, 4 * len(self.tasks)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_2d(self.task_axes)
        # self.task_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        for task_index, task in enumerate(self.tasks):

            
            task_fitness = self._get_task_fitness(task_index=task_index)
            
            task_vmin, task_vmax = np.nanpercentile(task_fitness, q=[1, 99])
            task_norm = mpl.colors.Normalize(task_vmin, task_vmax)

            # if task["transform"] == "log":
            #     task_norm = mpl.colors.LogNorm(task_vmin, task_vmax)
            # else:

            self.task_axes[task_index, 0].set_ylabel(f'{task["key"]}_fitness')

            self.task_axes[task_index, 0].set_title("samples")
            self.task_axes[task_index, 1].set_title("posterior mean")
            self.task_axes[task_index, 2].set_title("posterior std. dev.")

            data_ax = self.task_axes[task_index, 0].scatter(
                *self.inputs.values.T[axes], s=size, c=task_fitness, norm=task_norm, cmap=cmap
            )

            x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)

            task_posterior = task["model"].posterior(x)
            task_mean = task_posterior.mean
            task_sigma = task_posterior.variance.sqrt()

            if gridded:
                if not x.ndim == 3:
                    raise ValueError()
                self.task_axes[task_index, 1].pcolormesh(
                    x[..., 0].detach().numpy(),
                    x[..., 1].detach().numpy(),
                    task_mean[..., 0].detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[task_index, 2].pcolormesh(
                    x[..., 0].detach().numpy(),
                    x[..., 1].detach().numpy(),
                    task_sigma[..., 0].detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                )

            else:
                self.task_axes[task_index, 1].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=task_mean[..., 0].detach().numpy(),
                    s=size,
                    norm=task_norm,
                    cmap=cmap,
                )
                sigma_ax = self.task_axes[task_index, 2].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=task_sigma[..., 0].detach().numpy(),
                    s=size,
                    cmap=cmap,
                )

            self.task_fig.colorbar(data_ax, ax=self.task_axes[task_index, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(sigma_ax, ax=self.task_axes[task_index, 2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[axes[1]]["limits"])

    def plot_acquisition(self, acq_funcs=["ei"], **kwargs):
        if self._len_subset_dofs(kind="active", mode="on") == 1:
            self._plot_acq_one_dof(acq_funcs=acq_funcs, **kwargs)

        else:
            self._plot_acq_many_dofs(acq_funcs=acq_funcs, **kwargs)

    def _plot_acq_one_dof(self, acq_funcs, lw=1e0, **kwargs):
        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acq_funcs),
            figsize=(4 * len(acq_funcs), 4),
            sharex=True,
            constrained_layout=True,
        )

        self.acq_axes = np.atleast_1d(self.acq_axes)

        for iacq_func, acq_func_identifier in enumerate(acq_funcs):
            color = DEFAULT_COLOR_LIST[iacq_func]

            acq_func, acq_func_meta = self.get_acquisition_function(acq_func_identifier, return_metadata=True)

            x = self.test_inputs_grid
            *input_shape, input_dim = x.shape
            obj = acq_func.forward(x.reshape(-1, 1, input_dim)).reshape(input_shape)

            if acq_func_identifier in ["ei", "pi"]:
                obj = obj.exp()

            self.acq_axes[iacq_func].set_title(acq_func_meta["name"])

            on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]
            self.acq_axes[iacq_func].plot(
                x[..., on_dofs_are_active_mask].squeeze(), obj.detach().numpy(), lw=lw, color=color
            )

            self.acq_axes[iacq_func].set_xlim(self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_acq_many_dofs(
        self, acq_funcs, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16, **kwargs
    ):
        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acq_funcs),
            figsize=(4 * len(acq_funcs), 4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        if gridded is None:
            gridded = self._len_subset_dofs(kind="active", mode="on") == 2

        self.acq_axes = np.atleast_1d(self.acq_axes)
        # self.acq_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)
        *input_shape, input_dim = x.shape

        for iacq_func, acq_func_identifier in enumerate(acq_funcs):
            acq_func, acq_func_meta = self.get_acquisition_function(acq_func_identifier, return_metadata=True)

            obj = acq_func.forward(x.reshape(-1, 1, input_dim)).reshape(input_shape)
            if acq_func_identifier in ["ei", "pi"]:
                obj = obj.exp()

            if gridded:
                self.acq_axes[iacq_func].set_title(acq_func_meta["name"])
                obj_ax = self.acq_axes[iacq_func].pcolormesh(
                    x[..., 0].detach().numpy(),
                    x[..., 1].detach().numpy(),
                    obj.detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacq_func], location="bottom", aspect=32, shrink=0.8)

            else:
                self.acq_axes[iacq_func].set_title(acq_func_meta["name"])
                obj_ax = self.acq_axes[iacq_func].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=obj.detach().numpy(),
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacq_func], location="bottom", aspect=32, shrink=0.8)

        for ax in self.acq_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[axes[1]]["limits"])

    def plot_validity(self, **kwargs):
        if self._len_subset_dofs(kind="active", mode="on") == 1:
            self._plot_valid_one_dof(**kwargs)

        else:
            self._plot_valid_many_dofs(**kwargs)

    def _plot_valid_one_dof(self, size=16, lw=1e0):
        self.valid_fig, self.valid_ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, constrained_layout=True)

        x = self.test_inputs_grid
        *input_shape, input_dim = x.shape
        constraint = self.classifier.probabilities(x.reshape(-1, 1, input_dim))[..., -1].reshape(input_shape)

        self.valid_ax.scatter(self.inputs.values, self.all_tasks_valid, s=size)

        on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]

        self.valid_ax.plot(x[..., on_dofs_are_active_mask].squeeze(), constraint.detach().numpy(), lw=lw)

        self.valid_ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_valid_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=16, gridded=None):
        self.valid_fig, self.valid_axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True
        )

        if gridded is None:
            gridded = self._len_subset_dofs(kind="active", mode="on") == 2

        data_ax = self.valid_axes[0].scatter(
            *self.inputs.values.T[:2],
            c=self.all_tasks_valid,
            s=size,
            vmin=0,
            vmax=1,
            cmap=cmap,
        )

        x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)
        *input_shape, input_dim = x.shape
        constraint = self.classifier.probabilities(x.reshape(-1, 1, input_dim))[..., -1].reshape(input_shape)

        if gridded:
            self.valid_axes[1].pcolormesh(
                x[..., 0].detach().numpy(),
                x[..., 1].detach().numpy(),
                constraint.detach().numpy(),
                shading=shading,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

            # self.acq_fig.colorbar(obj_ax, ax=self.valid_axes[iacq_func], location="bottom", aspect=32, shrink=0.8)

        else:
            # self.valid_axes.set_title(acq_func_meta["name"])
            self.valid_axes[1].scatter(
                x.detach().numpy()[..., axes[0]],
                x.detach().numpy()[..., axes[1]],
                c=constraint.detach().numpy(),
            )

        self.valid_fig.colorbar(data_ax, ax=self.valid_axes[:2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.valid_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[axes[1]]["limits"])

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
            num_task_plots = self.num_tasks + 1

        self.num_tasks + 1 if self.num_tasks > 1 else 1

        hist_fig, hist_axes = plt.subplots(
            num_task_plots, 1, figsize=(6, 4 * num_task_plots), sharex=True, constrained_layout=True, dpi=200
        )
        hist_axes = np.atleast_1d(hist_axes)

        unique_strategies, acq_func_index, acq_func_inverse = np.unique(
            self.table.acq_func, return_index=True, return_inverse=True
        )

        sample_colors = np.array(DEFAULT_COLOR_LIST)[acq_func_inverse]

        if show_all_tasks:
            for task_index, task in enumerate(self.tasks):
                y = self.table.loc[:, f'{task["key"]}_fitness'].values
                hist_axes[task_index].scatter(x, y, c=sample_colors)
                hist_axes[task_index].plot(x, y, lw=5e-1, c="k")
                hist_axes[task_index].set_ylabel(task["key"])

        y = self.scalarized_fitness

        cummax_y = np.array([np.nanmax(y[: i + 1]) for i in range(len(y))])

        hist_axes[-1].scatter(x, y, c=sample_colors)
        hist_axes[-1].plot(x, y, lw=5e-1, c="k")

        hist_axes[-1].plot(x, cummax_y, lw=5e-1, c="k", ls=":")

        hist_axes[-1].set_ylabel("total_fitness")
        hist_axes[-1].set_xlabel(x_key)

        handles = []
        for i_acq_func, acq_func in enumerate(unique_strategies):
            #        i_acq_func = np.argsort(acq_func_index)[i_handle]
            handles.append(Patch(color=DEFAULT_COLOR_LIST[i_acq_func], label=acq_func))
        legend = hist_axes[0].legend(handles=handles, fontsize=8)
        legend.set_title("acquisition function")

    # plot_history(self, x_key="time")

    # plt.savefig("bo-history.pdf", dpi=256)
