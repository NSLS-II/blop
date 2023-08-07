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
from . import models

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
DEFAULT_COLORMAP = "turbo"


def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


def default_digestion_plan(db, uid):
    return db[uid].table(fill=True)


MAX_TEST_INPUTS = 2**11


TASK_CONFIG = {}

ACQF_CONFIG = {
    "quasi-random": {
        "identifiers": ["qr", "quasi-random"],
        "pretty_name": "Quasi-random",
        "description": "Sobol-sampled quasi-random points.",
    },
    "expected_mean": {
        "identifiers": ["em", "expected_mean"],
        "pretty_name": "Expected mean",
        "description": "The expected value at each input.",
    },
    "expected_improvement": {
        "identifiers": ["ei", "expected_improvement"],
        "pretty_name": "Expected improvement",
        "description": r"The expected value of max(f(x) - \nu, 0), where \nu is the current maximum.",
    },
    "probability_of_improvement": {
        "identifiers": ["pi", "probability_of_improvement"],
        "pretty_name": "Probability of improvement",
        "description": "The probability that this input improves on the current maximum.",
    },
    "upper_confidence_bound": {
        "identifiers": ["ucb", "upper_confidence_bound"],
        "default_args": {"z": 2},
        "pretty_name": "Upper confidence bound",
        "description": r"The expected value, plus some multiple of the uncertainty (typically \mu + 2\sigma).",
    },
}

TASK_TRANSFORMS = {"log": lambda x: np.log(x)}


def _validate_and_prepare_dofs(dofs):
    for dof in dofs:
        if type(dof) is not dict:
            raise ValueError("Supplied dofs must be a list of dicts!")
        if "device" not in dof.keys():
            raise ValueError("Each DOF must have a device!")

        dof["device"].kind = "hinted"

        if "limits" not in dof.keys():
            dof["limits"] = (-np.inf, np.inf)
        dof["limits"] = tuple(np.array(dof["limits"]).astype(float))

        # read-only DOFs (without a set method) are passive by default
        dof["kind"] = dof.get("kind", "active" if hasattr(dof["device"], "set") else "passive")
        if dof["kind"] not in ["active", "passive"]:
            raise ValueError('DOF kinds must be one of "active" or "passive"')

        dof["mode"] = dof.get("mode", "on" if dof["kind"] == "active" else "off")
        if dof["mode"] not in ["on", "off"]:
            raise ValueError('DOF modes must be one of "on" or "off"')

    dof_names = [dof["device"].name for dof in dofs]
    if not len(set(dof_names)) == len(dof_names):
        raise ValueError("Names of DOFs must be unique!")

    return list(dofs)


def _validate_and_prepare_tasks(tasks):
    for task in tasks:
        if type(task) is not dict:
            raise ValueError("Supplied tasks must be a list of dicts!")
        if task["kind"] not in ["minimize", "maximize"]:
            raise ValueError('"mode" must be specified as either "minimize" or "maximize"')
        if "weight" not in task.keys():
            task["weight"] = 1
        if "limits" not in task.keys():
            task["limits"] = (-np.inf, np.inf)

    task_keys = [task["key"] for task in tasks]
    if not len(set(task_keys)) == len(task_keys):
        raise ValueError("Keys of tasks must be unique!")

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
        self.digestion = kwargs.get("digestion", default_digestion_plan)
        self.dets = list(np.atleast_1d(kwargs.get("dets", [])))

        self.acqf_config = kwargs.get("acqf_config", ACQF_CONFIG)

        self.table = pd.DataFrame()

        self._initialized = False
        self._train_models = True
        self.a_priori_hypers = None

    def _subset_inputs_sampler(self, kind=None, mode=None, n=MAX_TEST_INPUTS):
        """
        Returns $n$ quasi-randomly sampled inputs in the bounded parameter space
        """
        transform = self._subset_input_transform(kind=kind, mode=mode)
        return transform.untransform(utils.normalized_sobol_sampler(n, d=self._n_subset_dofs(kind=kind, mode=mode)))

    def initialize(
        self,
        acqf=None,
        n_init=4,
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
        Inform the agent about new inputs and targets for the model.
        """

        new_table = pd.DataFrame() if new_table is None else new_table
        self.table = pd.concat([self.table, new_table]) if append else new_table
        self.table.index = np.arange(len(self.table))

        fitnesses = self.task_fitnesses  # computes from self.table

        # update fitness estimates
        self.table.loc[:, fitnesses.columns] = fitnesses.values
        self.table.loc[:, "total_fitness"] = fitnesses.values.sum(axis=1)

        skew_dims = [tuple(np.arange(self._n_subset_dofs(mode="on")))]

        if self._initialized:
            cached_hypers = self.hypers

        feasibility = ~fitnesses.isna().any(axis=1)

        if not feasibility.sum() >= 2:
            raise ValueError("There must be at least two feasible data points per task!")

        inputs = self.inputs.loc[feasibility, self._subset_dof_names(mode="on")].values
        train_inputs = torch.tensor(inputs).double()  # .unsqueeze(0)

        for task in self.tasks:
            targets = self.table.loc[feasibility, f'{task["key"]}_fitness'].values
            train_targets = torch.tensor(targets).double().unsqueeze(-1)  # .unsqueeze(0)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(1e-6).square(),
                    torch.tensor(1e-2).square(),
                ),
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

        # this ensures that we have equal weight between task fitness and feasibility fitness
        self.task_scalarization = botorch.acquisition.objective.ScalarizedPosteriorTransform(
            weights=torch.tensor([*torch.ones(self.n_tasks), self.fitness_variance.sum().sqrt()]).double(),
            offset=0,
        )

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            torch.tensor(feasibility).long(), learn_additional_noise=True
        ).double()

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=torch.tensor(self.inputs.values).double(),
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

        feasibility_fitness_model = botorch.models.deterministic.GenericDeterministicModel(
            f=lambda X: -self.classifier.log_prob(X).square()
        )

        self.model_list = botorch.models.model.ModelList(*[task["model"] for task in self.tasks], feasibility_fitness_model)

    @property
    def task_fitnesses(self):
        df = pd.DataFrame(index=self.table.index)
        for task in self.tasks:
            name = f'{task["key"]}_fitness'

            df.loc[:, name] = task["weight"] * self.table.loc[:, task["key"]]

            # check that task values are inside acceptable values
            valid = (df.loc[:, name] > task["limits"][0]) & (df.loc[:, name] < task["limits"][1])

            # transform if needed
            if "transform" in task.keys():
                if task["transform"] == "log":
                    valid &= df.loc[:, name] > 0
                    df.loc[valid, name] = np.log(df.loc[valid, name])
                    df.loc[~valid, name] = np.nan

            if task["kind"] == "minimize":
                df.loc[valid, name] *= -1
        return df

    def _dof_kind_mask(self, kind=None):
        return [dof["kind"] == kind if kind is not None else True for dof in self.dofs]

    def _dof_mode_mask(self, mode=None):
        return [dof["mode"] == mode if mode is not None else True for dof in self.dofs]

    def _dof_mask(self, kind=None, mode=None):
        return [(k and m) for k, m in zip(self._dof_kind_mask(kind), self._dof_mode_mask(mode))]

    def _subset_dofs(self, kind=None, mode=None):
        return [dof for dof, m in zip(self.dofs, self._dof_mask(kind, mode)) if m]

    def _n_subset_dofs(self, kind=None, mode=None):
        return len(self._subset_dofs(kind, mode))

    def _subset_devices(self, kind=None, mode=None):
        return [dof["device"] for dof in self._subset_dofs(kind, mode)]

    def _read_subset_devices(self, kind=None, mode=None):
        return [device.read()[device.name]["value"] for device in self._subset_devices(kind, mode)]

    def _subset_dof_names(self, kind=None, mode=None):
        return [device.name for device in self._subset_devices(kind, mode)]

    def _subset_dof_limits(self, kind=None, mode=None):
        dofs_subset = self._subset_dofs(kind, mode)
        if len(dofs_subset) > 0:
            return torch.tensor([dof["limits"] for dof in dofs_subset], dtype=torch.float64).T
        return torch.empty((2, 0))

    def test_inputs(self, n=MAX_TEST_INPUTS):
        return utils.sobol_sampler(self._acqf_bounds, n=n)

    @property
    def test_inputs_grid(self):
        n_side = int(MAX_TEST_INPUTS ** (1 / self._n_subset_dofs(kind="active", mode="on")))
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
    def _acqf_bounds(self):
        return torch.tensor(
            [
                dof["limits"] if dof["kind"] == "active" else tuple(2 * [dof["device"].read()[dof["device"].name]["value"]])
                for dof in self.dofs
                if dof["mode"] == "on"
            ]
        ).T

    @property
    def n_tasks(self):
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

    def _subset_input_transform(self, kind=None, mode=None):
        limits = self._subset_dof_limits(kind, mode)
        offset = limits.min(dim=0).values
        coefficient = limits.max(dim=0).values - offset
        return botorch.models.transforms.input.AffineInputTransform(
            d=limits.shape[-1], coefficient=coefficient, offset=offset
        )

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
        return sp.stats.qmc.Sobol(d=self._n_subset_dofs(kind="active", mode="on"), scramble=True).random(n=min_power_of_two)[
            subset
        ]

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
    def acqf_info(self):
        entries = []
        for k, d in self.acqf_config.items():
            ret = ""
            ret += f'{d["pretty_name"].upper()} (identifiers: {d["identifiers"]})\n'
            ret += f'-> {d["description"]}'
            entries.append(ret)

        print("\n\n".join(entries))

    def get_acquisition_function(self, acqf_identifier="ei", return_metadata=False, acqf_args={}, **kwargs):
        if not self._initialized:
            raise RuntimeError(f'Can\'t construct acquisition function "{acqf_identifier}" (the agent is not initialized!)')

        if acqf_identifier.lower() in ACQF_CONFIG["expected_improvement"]["identifiers"]:
            acqf = botorch.acquisition.analytic.LogExpectedImprovement(
                self.model_list,
                best_f=self.scalarized_fitness.max(),
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "expected improvement", "args": {}}

        elif acqf_identifier.lower() in ACQF_CONFIG["probability_of_improvement"]["identifiers"]:
            acqf = botorch.acquisition.analytic.LogProbabilityOfImprovement(
                self.model_list,
                best_f=self.scalarized_fitness.max(),
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "probability of improvement", "args": {}}

        elif acqf_identifier.lower() in ACQF_CONFIG["expected_mean"]["identifiers"]:
            acqf = botorch.acquisition.analytic.UpperConfidenceBound(
                self.model_list,
                beta=0,
                posterior_transform=self.task_scalarization,
                **kwargs,
            )
            acqf_meta = {"name": "expected mean"}

        elif acqf_identifier.lower() in ACQF_CONFIG["upper_confidence_bound"]["identifiers"]:
            beta = ACQF_CONFIG["upper_confidence_bound"]["default_args"]["z"] ** 2
            acqf = botorch.acquisition.analytic.UpperConfidenceBound(
                self.model_list,
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
            active_X = self._subset_inputs_sampler(n=n, kind="active", mode="on").squeeze(1).numpy()
            acqf_meta = {"name": "quasi-random", "args": {}}

        elif n == 1:
            active_X, acqf_meta = self.ask_single(acqf_identifier, return_metadata=True)

        elif n > 1:
            active_x_list = []
            for i in range(n):
                active_x, acqf_meta = self.ask_single(acqf_identifier, return_metadata=True)
                active_x_list.append(active_x)

                if i < (n - 1):
                    x = np.c_[active_x, acqf_meta["passive_values"]]
                    task_samples = [task["model"].posterior(torch.tensor(x)).sample().item() for task in self.tasks]
                    fantasy_table = pd.DataFrame(
                        np.c_[active_x, acqf_meta["passive_values"], np.atleast_2d(task_samples)],
                        columns=[
                            *self._subset_dof_names(kind="active", mode="on"),
                            *self._subset_dof_names(kind="passive", mode="on"),
                            *self.task_keys,
                        ],
                    )
                    self.tell(fantasy_table, train=False)

            active_X = np.concatenate(active_x_list, axis=0)
            self.forget(self.table.index[-(n - 1) :])

            if route:
                active_X = active_X[utils.route(self._read_subset_devices(kind="active", mode="on"), active_X)]

        return (active_X, acqf_meta) if return_metadata else active_X

    def ask_single(
        self,
        acqf_identifier="ei",
        return_metadata=False,
    ):
        """
        The next $n$ points to sample, recommended by the self. Returns
        """

        t0 = ttime.monotonic()

        acqf, acqf_meta = self.get_acquisition_function(acqf_identifier=acqf_identifier, return_metadata=True)

        BATCH_SIZE = 1
        NUM_RESTARTS = 8
        RAW_SAMPLES = 256

        candidates, _ = botorch.optim.optimize_acqf(
            acq_function=acqf,
            bounds=self._acqf_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        )

        x = candidates.numpy().astype(float)

        active_x = x[..., [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]]
        passive_x = x[..., [dof["kind"] != "active" for dof in self._subset_dofs(mode="on")]]

        acqf_meta["passive_values"] = passive_x

        if self.verbose:
            print(f"found point {x} in {ttime.monotonic() - t0:.02f} seconds")

        return (active_x, acqf_meta) if return_metadata else active_x

    def acquire(self, active_inputs):
        """
        Acquire and digest according to the self's acquisition and digestion plans.

        This should yield a table of sampled tasks with the same length as the sampled inputs.
        """
        try:
            active_devices = self._subset_devices(kind="active", mode="on")
            passive_devices = [*self._subset_devices(kind="passive"), *self._subset_devices(kind="active", mode="off")]

            uid = yield from self.acquisition_plan(
                active_devices, active_inputs.astype(float), [*self.dets, *passive_devices]
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

    @property
    def inputs(self):
        return self.table.loc[:, self._subset_dof_names(mode="on")].astype(float)

    @property
    def fitness_variance(self):
        return torch.tensor(np.nanvar(self.task_fitnesses.values, axis=0))

    @property
    def scalarized_fitness(self):
        return self.task_fitnesses.sum(axis=1)

    # @property
    # def best_sum_of_tasks_inputs(self):
    #     return self.inputs[np.nanargmax(self.task_fitnesses.sum(axis=1))]

    @property
    def go_to(self, inputs):
        yield from bps.mv(*[_ for items in zip(self._subset_dofs(kind="active"), np.atleast_1d(inputs).T) for _ in items])

    # @property
    # def go_to_best_sum_of_tasks(self):
    #     yield from self.go_to(self.best_sum_of_tasks_inputs)

    def plot_tasks(self, **kwargs):
        if self._n_subset_dofs(kind="active", mode="on") == 1:
            self._plot_tasks_one_dof(**kwargs)
        else:
            self._plot_tasks_many_dofs(**kwargs)

    def _plot_tasks_one_dof(self, size=16, lw=1e0):
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

            self.task_axes[itask].set_ylabel(task["key"])

            x = self.test_inputs_grid
            task_posterior = task["model"].posterior(x)
            task_mean = task_posterior.mean.detach().numpy()
            task_sigma = task_posterior.variance.sqrt().detach().numpy()

            self.task_axes[itask].scatter(
                self.inputs.loc[:, self._subset_dof_names(kind="active", mode="on")],
                self.table.loc[:, f'{task["key"]}_fitness'],
                s=size,
                color=color,
            )

            on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]

            for z in [0, 1, 2]:
                self.task_axes[itask].fill_between(
                    x[..., on_dofs_are_active_mask].squeeze(),
                    (task_mean - z * task_sigma).squeeze(),
                    (task_mean + z * task_sigma).squeeze(),
                    lw=lw,
                    color=color,
                    alpha=0.5**z,
                )

            self.task_axes[itask].set_xlim(self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_tasks_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16):
        if gridded is None:
            gridded = self._n_subset_dofs(kind="active", mode="on") == 2

        self.task_fig, self.task_axes = plt.subplots(
            self.n_tasks,
            3,
            figsize=(10, 4 * self.n_tasks),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_2d(self.task_axes)
        # self.task_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        feasible = ~self.task_fitnesses.isna().any(axis=1)

        for itask, task in enumerate(self.tasks):
            sampled_fitness = np.where(feasible, self.table.loc[:, f'{task["key"]}_fitness'].values, np.nan)
            task_vmin, task_vmax = np.nanpercentile(sampled_fitness, q=[1, 99])
            task_norm = mpl.colors.Normalize(task_vmin, task_vmax)

            # if task["transform"] == "log":
            #     task_norm = mpl.colors.LogNorm(task_vmin, task_vmax)
            # else:

            self.task_axes[itask, 0].set_ylabel(f'{task["key"]}_fitness')

            self.task_axes[itask, 0].set_title("samples")
            self.task_axes[itask, 1].set_title("posterior mean")
            self.task_axes[itask, 2].set_title("posterior std. dev.")

            data_ax = self.task_axes[itask, 0].scatter(
                *self.inputs.values.T[axes], s=size, c=sampled_fitness, norm=task_norm, cmap=cmap
            )

            x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)

            task_posterior = task["model"].posterior(x)
            task_mean = task_posterior.mean
            task_sigma = task_posterior.variance.sqrt()

            if gridded:
                if not x.ndim == 3:
                    raise ValueError()
                self.task_axes[itask, 1].pcolormesh(
                    x[..., 0],
                    x[..., 1],
                    task_mean[..., 0].detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                    norm=task_norm,
                )
                sigma_ax = self.task_axes[itask, 2].pcolormesh(
                    x[..., 0],
                    x[..., 1],
                    task_sigma[..., 0].detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                )

            else:
                self.task_axes[itask, 1].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=task_mean[..., 0].detach().numpy(),
                    s=size,
                    norm=task_norm,
                    cmap=cmap,
                )
                sigma_ax = self.task_axes[itask, 2].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=task_sigma[..., 0].detach().numpy(),
                    s=size,
                    cmap=cmap,
                )

            self.task_fig.colorbar(data_ax, ax=self.task_axes[itask, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(sigma_ax, ax=self.task_axes[itask, 2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[axes[1]]["limits"])

    def plot_acquisition(self, acqfs=["ei"], **kwargs):
        if self._n_subset_dofs(kind="active", mode="on") == 1:
            self._plot_acq_one_dof(acqfs=acqfs, **kwargs)

        else:
            self._plot_acq_many_dofs(acqfs=acqfs, **kwargs)

    def _plot_acq_one_dof(self, acqfs, lw=1e0, **kwargs):
        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acqfs),
            figsize=(4 * len(acqfs), 4),
            sharex=True,
            constrained_layout=True,
        )

        self.acq_axes = np.atleast_1d(self.acq_axes)

        for iacqf, acqf_identifier in enumerate(acqfs):
            color = DEFAULT_COLOR_LIST[iacqf]

            acqf, acqf_meta = self.get_acquisition_function(acqf_identifier, return_metadata=True)

            x = self.test_inputs_grid
            *input_shape, input_dim = x.shape
            obj = acqf.forward(x.reshape(-1, 1, input_dim)).reshape(input_shape)

            if acqf_identifier in ["ei", "pi"]:
                obj = obj.exp()

            self.acq_axes[iacqf].set_title(acqf_meta["name"])

            on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]
            self.acq_axes[iacqf].plot(x[..., on_dofs_are_active_mask].squeeze(), obj.detach().numpy(), lw=lw, color=color)

            self.acq_axes[iacqf].set_xlim(self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_acq_many_dofs(
        self, acqfs, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16, **kwargs
    ):
        self.acq_fig, self.acq_axes = plt.subplots(
            1,
            len(acqfs),
            figsize=(4 * len(acqfs), 4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        if gridded is None:
            gridded = self._n_subset_dofs(kind="active", mode="on") == 2

        self.acq_axes = np.atleast_1d(self.acq_axes)
        # self.acq_fig.suptitle(f"(x,y)=({self.dofs[axes[0]].name},{self.dofs[axes[1]].name})")

        x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)
        *input_shape, input_dim = x.shape

        for iacqf, acqf_identifier in enumerate(acqfs):
            acqf, acqf_meta = self.get_acquisition_function(acqf_identifier, return_metadata=True)

            obj = acqf.forward(x.reshape(-1, 1, input_dim)).reshape(input_shape)
            if acqf_identifier in ["ei", "pi"]:
                obj = obj.exp()

            if gridded:
                self.acq_axes[iacqf].set_title(acqf_meta["name"])
                obj_ax = self.acq_axes[iacqf].pcolormesh(
                    x[..., 0],
                    x[..., 1],
                    obj.detach().numpy(),
                    shading=shading,
                    cmap=cmap,
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

            else:
                self.acq_axes[iacqf].set_title(acqf_meta["name"])
                obj_ax = self.acq_axes[iacqf].scatter(
                    x.detach().numpy()[..., axes[0]],
                    x.detach().numpy()[..., axes[1]],
                    c=obj.detach().numpy(),
                )

                self.acq_fig.colorbar(obj_ax, ax=self.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

        for ax in self.acq_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[axes[1]]["limits"])

    def plot_feasibility(self, **kwargs):
        if self._n_subset_dofs(kind="active", mode="on") == 1:
            self._plot_feas_one_dof(**kwargs)

        else:
            self._plot_feas_many_dofs(**kwargs)

    def _plot_feas_one_dof(self, size=16, lw=1e0):
        self.feas_fig, self.feas_ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, constrained_layout=True)

        x = self.test_inputs_grid
        *input_shape, input_dim = x.shape
        log_prob = self.classifier.log_prob(x.reshape(-1, 1, input_dim)).reshape(input_shape)

        self.feas_ax.scatter(self.inputs.values, ~self.task_fitnesses.isna().any(axis=1), s=size)

        on_dofs_are_active_mask = [dof["kind"] == "active" for dof in self._subset_dofs(mode="on")]

        self.feas_ax.plot(x[..., on_dofs_are_active_mask].squeeze(), log_prob.exp().detach().numpy(), lw=lw)

        self.feas_ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[0]["limits"])

    def _plot_feas_many_dofs(self, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=16, gridded=None):
        self.feas_fig, self.feas_axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True)

        if gridded is None:
            gridded = self._n_subset_dofs(kind="active", mode="on") == 2

        data_ax = self.feas_axes[0].scatter(
            *self.inputs.values.T[:2],
            c=~self.task_fitnesses.isna().any(axis=1),
            s=size,
            vmin=0,
            vmax=1,
            cmap=cmap,
        )

        x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=MAX_TEST_INPUTS)
        *input_shape, input_dim = x.shape
        log_prob = self.classifier.log_prob(x.reshape(-1, 1, input_dim)).reshape(input_shape)

        if gridded:
            self.feas_axes[1].pcolormesh(
                x[..., 0],
                x[..., 1],
                log_prob.exp().detach().numpy(),
                shading=shading,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

            # self.acq_fig.colorbar(obj_ax, ax=self.feas_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

        else:
            # self.feas_axes.set_title(acqf_meta["name"])
            self.feas_axes[1].scatter(
                x.detach().numpy()[..., axes[0]],
                x.detach().numpy()[..., axes[1]],
                c=log_prob.exp().detach().numpy(),
            )

        self.feas_fig.colorbar(data_ax, ax=self.feas_axes[:2], location="bottom", aspect=32, shrink=0.8)

        for ax in self.feas_axes.ravel():
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
            num_task_plots = self.n_tasks + 1

        self.n_tasks + 1 if self.n_tasks > 1 else 1

        hist_fig, hist_axes = plt.subplots(
            num_task_plots, 1, figsize=(6, 4 * num_task_plots), sharex=True, constrained_layout=True, dpi=200
        )
        hist_axes = np.atleast_1d(hist_axes)

        unique_strategies, acqf_index, acqf_inverse = np.unique(self.table.acqf, return_index=True, return_inverse=True)

        sample_colors = np.array(DEFAULT_COLOR_LIST)[acqf_inverse]

        if show_all_tasks:
            for itask, task in enumerate(self.tasks):
                y = self.table.loc[:, f'{task["key"]}_fitness'].values
                hist_axes[itask].scatter(x, y, c=sample_colors)
                hist_axes[itask].plot(x, y, lw=5e-1, c="k")
                hist_axes[itask].set_ylabel(task["key"])

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
