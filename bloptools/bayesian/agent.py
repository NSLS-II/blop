import logging
import os
import time as ttime
import warnings
from collections import OrderedDict

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import botorch
import gpytorch
import h5py
import IPython as ip
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP

from .. import utils
from . import acquisition, models, plotting
from .acquisition import default_acquisition_plan
from .devices import DOFList
from .digestion import default_digestion_function
from .objective import ObjectiveList

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")

MAX_TEST_INPUTS = 2**11


class Agent:
    def __init__(
        self,
        dofs,
        objectives,
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
        objectives : iterable of objectives
            The objectives which the agent will try to optimize.
        acquisition : Bluesky plan generator that takes arguments (dofs, inputs, dets)
            A plan that samples the beamline for some given inputs.
        digestion : function that takes arguments (db, uid)
            A function to digest the output of the acquisition.
        db : A databroker instance.
        """

        # DOFs are parametrized by whether they are active and whether they are read-only
        #
        # below are the behaviors of DOFs of each kind and mode:
        #
        # "read": the agent will read the input on every acquisition (all dofs are always read)
        # "move": the agent will try to set and optimize over these (there must be at least one of these)
        # "input" means that the agent will use the value to make its posterior
        #
        #
        #               not read-only        read-only
        #          +---------------------+---------------+
        #   active |  read, input, move  |  read, input  |
        #          +---------------------+---------------+
        # inactive |  read               |  read         |
        #          +---------------------+---------------+
        #
        #

        self.dofs = DOFList(list(np.atleast_1d(dofs)))
        self.objectives = ObjectiveList(list(np.atleast_1d(objectives)))
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

        self.initialized = False
        self._train_models = True
        self.a_priori_hypers = None

        self.plots = {"objectives": {}}

    def tell(self, new_table=None, append=True, train=True, **kwargs):
        """
        Inform the agent about new inputs and targets for the model.

        If run with no arguments, it will just reconstruct all the models.
        """

        new_table = pd.DataFrame() if new_table is None else new_table
        self.table = pd.concat([self.table, new_table]) if append else new_table
        self.table.index = np.arange(len(self.table))

        skew_dims = self.latent_dim_tuples

        if self.initialized:
            cached_hypers = self.hypers

        inputs = self.table.loc[:, self.dofs.subset(active=True).names].values.astype(float)

        for i, obj in enumerate(self.objectives):
            self.table.loc[:, f"{obj.key}_fitness"] = targets = self._get_objective_targets(i)
            train_index = ~np.isnan(targets)

            if not train_index.sum() >= 2:
                raise ValueError("There must be at least two valid data points per objective!")

            train_inputs = torch.tensor(inputs[train_index], dtype=torch.double)
            train_targets = torch.tensor(targets[train_index], dtype=torch.double).unsqueeze(-1)  # .unsqueeze(0)

            # for constructing the log normal noise prior
            # target_snr = 2e2
            # scale = 2e0
            # loc = np.log(1 / target_snr**2) + scale**2

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(
                    torch.tensor(1e-4).square(),
                    torch.tensor(1 / obj.min_snr).square(),
                ),
                # noise_prior=gpytorch.priors.torch_priors.LogNormalPrior(loc=loc, scale=scale),
            ).double()

            outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)  # , batch_shape=torch.Size((1,)))

            obj.model = models.LatentGP(
                train_inputs=train_inputs,
                train_targets=train_targets,
                likelihood=likelihood,
                skew_dims=skew_dims,
                input_transform=self._subset_input_transform(active=True),
                outcome_transform=outcome_transform,
            ).double()

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            self.all_objectives_valid.long(), learn_additional_noise=True
        ).double()

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=torch.tensor(inputs).double(),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            skew_dims=skew_dims,
            likelihood=dirichlet_likelihood,
            input_transform=self._subset_input_transform(active=True),
        ).double()

        if self.a_priori_hypers is not None:
            self._set_hypers(self.a_priori_hypers)
        elif not train:
            self._set_hypers(cached_hypers)
        else:
            try:
                self.train_models()
            except botorch.exceptions.errors.ModelFittingError:
                if self.initialized:
                    self._set_hypers(cached_hypers)
                else:
                    raise RuntimeError("Could not fit model on initialization!")

        self.constraint = GenericDeterministicModel(f=lambda x: self.classifier.probabilities(x)[..., -1])

    def ask(self, acq_func_identifier="qei", n=1, route=True, sequential=True, **acq_func_kwargs):
        """
        Ask the agent for the best point to sample, given an acquisition function.

        acq_func_identifier: which acquisition function to use
        n: how many points to get
        """

        acq_func_name = acquisition.parse_acq_func(acq_func_identifier)
        acq_func_type = acquisition.config[acq_func_name]["type"]

        start_time = ttime.monotonic()

        if self.verbose:
            print(f'finding points with acquisition function "{acq_func_name}" ...')

        if acq_func_type in ["analytic", "monte_carlo"]:
            if not self.initialized:
                raise RuntimeError(
                    f'Can\'t construct non-trivial acquisition function "{acq_func_identifier}"'
                    f" (the agent is not initialized!)"
                )

            if acq_func_type == "analytic" and n > 1:
                raise ValueError("Can't generate multiple design points for analytic acquisition functions.")

            acq_func, acq_func_meta = self.get_acquisition_function(
                acq_func_identifier=acq_func_identifier, return_metadata=True
            )

            NUM_RESTARTS = 8
            RAW_SAMPLES = 1024

            candidates, acqf_objective = botorch.optim.optimize_acqf(
                acq_function=acq_func,
                bounds=self.acquisition_function_bounds,
                q=n,
                sequential=sequential,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            )

            x = candidates.numpy().astype(float)

            active_dofs_are_read_only = np.array([dof.read_only for dof in self.dofs.subset(active=True)])

            acquisition_X = x[..., ~active_dofs_are_read_only]
            read_only_X = x[..., active_dofs_are_read_only]
            acq_func_meta["read_only_values"] = read_only_X

        else:
            if acq_func_name == "random":
                acquisition_X = torch.rand()
                acq_func_meta = {"name": "random", "args": {}}

            if acq_func_name == "quasi-random":
                acquisition_X = self._subset_inputs_sampler(n=n, active=True, read_only=False).squeeze(1).numpy()
                acq_func_meta = {"name": "quasi-random", "args": {}}

            elif acq_func_name == "grid":
                n_active_dims = len(self.dofs.subset(active=True, read_only=False))
                acquisition_X = self.test_inputs_grid(max_inputs=n).reshape(-1, n_active_dims).numpy()
                acq_func_meta = {"name": "grid", "args": {}}

            else:
                raise ValueError()

            # define dummy acqf objective
            acqf_objective = None

        acq_func_meta["duration"] = duration = ttime.monotonic() - start_time

        if self.verbose:
            summary = pd.DataFrame(acquisition_X, columns=self.dofs.subset(active=True, read_only=False).names)
            summary.insert(0, "acqf", acqf_objective)

            print(f"found points in {duration:.03f} seconds:\n" + summary.__repr__())

        if route and n > 1:
            routing_index = utils.route(self.dofs.subset(active=True, read_only=False).readback, acquisition_X)
            acquisition_X = acquisition_X[routing_index]

        return acquisition_X, acq_func_meta

    def acquire(self, acquisition_inputs):
        """
        Acquire and digest according to the self's acquisition and digestion plans.

        This should yield a table of sampled objectives with the same length as the sampled inputs.
        """
        try:
            acquisition_devices = self.dofs.subset(active=True, read_only=False).devices
            # read_only_devices = self.dofs.subset(active=True, read_only=True).devices

            # the acquisition plan always takes as arguments:
            # (things to move, where to move them, things to trigger once you get there)
            # with some other kwargs

            uid = yield from self.acquisition_plan(
                acquisition_devices,
                acquisition_inputs.astype(float),
                [*self.dets, *self.dofs.devices],
                delay=self.trigger_delay,
            )

            products = self.digestion(self.db, uid)

            # compute the fitness for each objective
            # for index, entry in products.iterrows():
            #     for obj in self.objectives:
            #         products.loc[index, objective["key"]] = getattr(entry, objective["key"])

        except Exception as error:
            if not self.allow_acquisition_errors:
                raise error
            logging.warning(f"Error in acquisition/digestion: {repr(error)}")
            products = pd.DataFrame(acquisition_inputs, columns=self.dofs.subset(active=True, read_only=False).names)
            for obj in self.objectives:
                products.loc[:, obj.key] = np.nan

        if not len(acquisition_inputs) == len(products):
            raise ValueError("The table returned by the digestion function must be the same length as the sampled inputs!")

        return products

    def learn(
        self,
        acq_func=None,
        n=1,
        iterations=1,
        upsample=1,
        train=True,
        data=None,
        **kwargs,
    ):
        """
        This iterates the learning algorithm, looping over ask -> acquire -> tell.
        It should be passed to a Bluesky RunEngine.
        """

        if data is not None:
            if type(data) == str:
                self.tell(new_table=pd.read_hdf(data, key="table"))
            else:
                self.tell(new_table=data)

        if self.sample_center_on_init and not self.initialized:
            new_table = yield from self.acquire(self.dofs.subset(active=True, read_only=False).limits.mean(axis=1))
            new_table.loc[:, "acq_func"] = "sample_center_on_init"
            self.tell(new_table=new_table, train=False)

        if acq_func is not None:
            for i in range(iterations):
                x, acq_func_meta = self.ask(n=n, acq_func_identifier=acq_func, **kwargs)

                new_table = yield from self.acquire(x)
                new_table.loc[:, "acq_func"] = acq_func_meta["name"]
                self.tell(new_table=new_table, train=train)

        self.initialized = True

    def get_acquisition_function(self, acq_func_identifier, return_metadata=False):
        return acquisition.get_acquisition_function(
            self, acq_func_identifier=acq_func_identifier, return_metadata=return_metadata
        )

    def reset(self):
        """
        Reset the agent.
        """
        self.table = pd.DataFrame()
        self.initialized = False

    def benchmark(
        self, output_dir="./", runs=16, n_init=64, learning_kwargs_list=[{"acq_func": "qei", "n": 4, "iterations": 16}]
    ):
        cache_limits = {dof.name: dof.limits for dof in self.dofs}

        for run in range(runs):
            for dof in self.dofs:
                offset = 0.25 * np.ptp(dof.limits) * np.random.uniform(low=-1, high=1)
                dof.limits = (cache_limits[dof.name][0] + offset, cache_limits[dof.name][1] + offset)

            self.reset()

            yield from self.learn("qr", n=n_init)

            for kwargs in learning_kwargs_list:
                yield from self.learn(**kwargs)

            self.save_data(output_dir + f"benchmark-{int(ttime.time())}.h5")

            ip.display.clear_output(wait=True)

    @property
    def model(self):
        """
        A model encompassing all the objectives. A single GP in the single-objective case, or a model list.
        """
        return ModelListGP(*[obj.model for obj in self.objectives]) if len(self.objectives) > 1 else self.objectives[0].model

    @property
    def objective_weights_torch(self):
        return torch.tensor(self.objectives.weights, dtype=torch.double)

    def _get_objective_targets(self, i):
        """
        Returns the targets (what we fit to) for an objective, given the objective index.
        """
        obj = self.objectives[i]

        targets = self.table.loc[:, obj.key].values.copy()

        # check that targets values are inside acceptable values
        valid = (targets > obj.limits[0]) & (targets < obj.limits[1])
        targets = np.where(valid, targets, np.nan)

        # transform if needed
        if obj.log:
            targets = np.where(valid, np.log(targets), np.nan)

        if obj.minimize:
            targets *= -1

        return targets

    @property
    def n_objs(self):
        """
        Returns a (num_objectives x n_observations) array of objectives
        """
        return len(self.objectives)

    @property
    def objectives_targets(self):
        """
        Returns a (num_objectives x n_obs) array of objectives
        """
        return torch.cat([torch.tensor(self._get_objective_targets(i))[..., None] for i in range(self.n_objs)], dim=1)

    @property
    def scalarized_objectives(self):
        return (self.objectives_targets * self.objectives.weights).sum(axis=-1)

    @property
    def best_scalarized_objective(self):
        f = self.scalarized_objectives
        return np.where(np.isnan(f), -np.inf, f).max()

    @property
    def all_objectives_valid(self):
        return ~torch.isnan(self.scalarized_objectives)

    @property
    def target_names(self):
        return [f"{obj.key}_fitness" for obj in self.objectives]

    def test_inputs_grid(self, max_inputs=MAX_TEST_INPUTS):
        """
        Returns a (n_side, ..., n_side, 1, n_active_dof) grid of test_inputs.
        n_side is 1 if a dof is read-only
        """
        n_settable_acq_func_dofs = len(self.dofs.subset(active=True, read_only=False))
        n_side_settable = int(np.power(max_inputs, n_settable_acq_func_dofs**-1))
        n_sides = [1 if dof.read_only else n_side_settable for dof in self.dofs.subset(active=True)]
        return torch.cat(
            [
                tensor.unsqueeze(-1)
                for tensor in torch.meshgrid(
                    *[
                        torch.linspace(lower_limit, upper_limit, n_side)
                        for (lower_limit, upper_limit), n_side in zip(self.dofs.subset(active=True).limits, n_sides)
                    ],
                    indexing="ij",
                )
            ],
            dim=-1,
        ).unsqueeze(-2)

    def test_inputs(self, n=MAX_TEST_INPUTS):
        """
        Returns a (n, 1, n_active_dof) grid of test_inputs
        """
        return utils.sobol_sampler(self.acquisition_function_bounds, n=n)

    @property
    def acquisition_function_bounds(self):
        """
        Returns a (2, n_active_dof) array of bounds for the acquisition function
        """
<<<<<<< HEAD
        active_dofs = self.dofs.subset(active=True)

        acq_func_lower_bounds = [dof.lower_limit if not dof.read_only else dof.readback for dof in active_dofs]
        acq_func_upper_bounds = [dof.upper_limit if not dof.read_only else dof.readback for dof in active_dofs]
=======
        acq_func_lower_bounds = [dof.lower_limit if not dof.read_only else dof.readback for dof in self.dofs]
        acq_func_upper_bounds = [dof.upper_limit if not dof.read_only else dof.readback for dof in self.dofs]
>>>>>>> 39a579f (make sure DOF bounds are cast to floats)

        return torch.tensor(np.vstack([acq_func_lower_bounds, acq_func_upper_bounds]), dtype=torch.double)

        return torch.tensor(
            [dof.limits if not dof.read_only else tuple(2 * [dof.readback]) for dof in self.dofs.subset(active=True)]
        ).T

    # @property
    # def num_objectives(self):
    #     return len(self.objectives)

    # @property
    # def det_names(self):
    #     return [det.name for det in self.dets]

    # @property
    # def objective_keys(self):
    #     return [obj.key for obj in self.objectives]

    # @property
    # def objective_models(self):
    #     return [obj.model for obj in self.objectives]

    # @property
    # def objective_weights(self):
    #     return torch.tensor([objective["weight"] for obj in self.objectives], dtype=torch.float64)

    # @property
    # def objective_signs(self):
    #     return torch.tensor([(-1 if objective["minimize"] else +1) for obj in self.objectives], dtype=torch.long)

    @property
    def latent_dim_tuples(self):
        """
        Returns a list of tuples, where each tuple represent a group of dimension to find a latent representation of.
        """

        latent_dim_labels = [dof.latent_group for dof in self.dofs.subset(active=True)]
        u, uinv = np.unique(latent_dim_labels, return_inverse=True)

        return [tuple(np.where(uinv == i)[0]) for i in range(len(u))]

    def _subset_input_transform(self, active=None, read_only=None, tags=[]):
        # torch likes limits to be (2, n_dof) and not (n_dof, 2)
        torch_limits = torch.tensor(self.dofs.subset(active, read_only, tags).limits.T, dtype=torch.double)
        offset = torch_limits.min(dim=0).values
        coefficient = torch_limits.max(dim=0).values - offset
        return botorch.models.transforms.input.AffineInputTransform(
            d=torch_limits.shape[-1], coefficient=coefficient, offset=offset
        )

    def _subset_inputs_sampler(self, active=None, read_only=None, tags=[], n=MAX_TEST_INPUTS):
        """
        Returns $n$ quasi-randomly sampled inputs in the bounded parameter space
        """
        transform = self._subset_input_transform(active, read_only, tags)
        return transform.untransform(utils.normalized_sobol_sampler(n, d=len(self.dofs.subset(active, read_only, tags))))

    def save_data(self, filepath="./self_data.h5"):
        """
        Save the sampled inputs and targets of the agent to a file, which can be used
        to initialize a future self.
        """

        self.table.to_hdf(filepath, key="table")

    def forget(self, index):
        self.tell(new_table=self.table.drop(index=index), append=False, train=False)

    def sampler(self, n, d):
        """
        Returns $n$ quasi-randomly sampled points on the [0,1] ^ d hypercube using Sobol sampling.
        """
        min_power_of_two = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        subset = np.random.choice(min_power_of_two, size=n, replace=False)
        return sp.stats.qmc.Sobol(d=d, scramble=True).random(n=min_power_of_two)[subset]

    def _set_hypers(self, hypers):
        for obj in self.objectives:
            obj.model.load_state_dict(hypers[obj.key])
        self.classifier.load_state_dict(hypers["classifier"])

    @property
    def hypers(self):
        hypers = {"classifier": {}}
        for key, value in self.classifier.state_dict().items():
            hypers["classifier"][key] = value
        for obj in self.objectives:
            hypers[obj.key] = {}
            for key, value in obj.model.state_dict().items():
                hypers[obj.key][key] = value

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
        for obj in self.objectives:
            model = obj.model
            botorch.fit.fit_gpytorch_mll(gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model), **kwargs)
        botorch.fit.fit_gpytorch_mll(
            gpytorch.mlls.ExactMarginalLogLikelihood(self.classifier.likelihood, self.classifier), **kwargs
        )
        if self.verbose:
            print(f"trained models in {ttime.monotonic() - t0:.01f} seconds")

    @property
    def acq_func_info(self):
        entries = []
        for k, d in self.acq_func_config.items():
            ret = ""
            ret += f'{d["pretty_name"].upper()} (identifiers: {d["identifiers"]})\n'
            ret += f'-> {d["description"]}'
            entries.append(ret)

        print("\n\n".join(entries))

    @property
    def inputs(self):
        return self.table.loc[:, self.dofs.device_names].astype(float)

    @property
    def active_inputs(self):
        return self.table.loc[:, self.dofs.subset(active=True).device_names].astype(float)

    @property
    def acquisition_inputs(self):
        return self.table.loc[:, self.dofs.subset(active=True, read_only=False).names].astype(float)

    @property
    def best_inputs(self):
        """
        Returns a value for each currently active and non-read-only degree of freedom
        """
        return self.table.loc[np.nanargmax(self.scalarized_objectives), self.dofs.subset(active=True, read_only=False).names]

    def go_to(self, positions):
        args = []
        for dof, value in zip(self.dofs.subset(active=True, read_only=False), np.atleast_1d(positions)):
            args.append(dof.device)
            args.append(value)
        yield from bps.mv(*args)

    def go_to_best(self):
        yield from self.go_to(self.best_inputs)

    def plot_objectives(self, **kwargs):
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_objs_one_dof(self, **kwargs)
        else:
            plotting._plot_objs_many_dofs(self, **kwargs)

    def plot_acquisition(self, acq_funcs=["ei"], **kwargs):
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_acq_one_dof(self, acq_funcs=acq_funcs, **kwargs)

        else:
            plotting._plot_acq_many_dofs(self, acq_funcs=acq_funcs, **kwargs)

    def plot_validity(self, **kwargs):
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_valid_one_dof(self, **kwargs)

        else:
            plotting._plot_valid_many_dofs(self, **kwargs)

    def plot_history(self, **kwargs):
<<<<<<< HEAD
        plotting._plot_history(self, **kwargs)
=======
        plotting._plot_history(self, **kwargs)
>>>>>>> 39a579f (make sure DOF bounds are cast to floats)
