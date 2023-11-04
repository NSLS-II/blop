import logging
import time as ttime
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Callable, Sequence, Tuple

import bluesky.plan_stubs as bps  # noqa F401
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
from databroker import Broker
from ophyd import Signal

from .. import utils
from . import acquisition, models, plotting
from .acquisition import default_acquisition_plan
from .devices import DOF, DOFList
from .digestion import default_digestion_function
from .objective import Objective, ObjectiveList

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")

MAX_TEST_INPUTS = 2**11


class Agent:
    def __init__(
        self,
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        db: Broker = None,
        dets: Sequence[Signal] = [],
        acquistion_plan=default_acquisition_plan,
        digestion: Callable = default_digestion_function,
        verbose: bool = False,
        tolerate_acquisition_errors=False,
        sample_center_on_init=False,
        trigger_delay: float = 0,
    ):
        """
        A Bayesian optimization agent.

        Parameters
        ----------
        dofs : iterable of DOF objects
            The degrees of freedom that the agent can control, which determine the output of the model.
        objectives : iterable of Objective objects
            The objectives which the agent will try to optimize.
        dets : iterable of ophyd objects
            Detectors to trigger during acquisition.
        acquisition_plan : optional
            A plan that samples the beamline for some given inputs.
        digestion :
            A function to digest the output of the acquisition, taking arguments (db, uid).
        db : optional
            A databroker instance.
        verbose : bool
            To be verbose or not.
        tolerate_acquisition_errors : bool
            Whether to allow errors during acquistion. If `True`, errors will be caught as warnings.
        sample_center_on_init : bool
            Whether to sample the center of the DOF limits when the agent has no data yet.
        trigger_delay : float
            How many seconds to wait between moving DOFs and triggering detectors.
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

        self.dets = dets
        self.acquisition_plan = acquistion_plan
        self.digestion = digestion

        self.verbose = verbose

        self.tolerate_acquisition_errors = tolerate_acquisition_errors

        self.trigger_delay = trigger_delay
        self.sample_center_on_init = sample_center_on_init

        self.table = pd.DataFrame()

        self.initialized = False
        self.a_priori_hypers = None

    def tell(self, x: Mapping, y: Mapping, metadata=None, append=True, train_models=True, hypers=None):
        """
        Inform the agent about new inputs and targets for the model.

        If run with no arguments, it will just reconstruct all the models.

        Parameters
        ----------
        x : dict
            A dict keyed by the name of each DOF, with a list of values for each DOF.
        y : dict
            A dict keyed by the name of each objective, with a list of values for each objective.
        append: bool
            If `True`, will append new data to old data. If `False`, will replace old data with new data.
        train_models: bool
            Whether to train the models on construction.
        hypers:
            A dict of hyperparameters for the model to assume a priori.
        """

        new_table = pd.DataFrame({**x, **y, **metadata} if metadata is not None else {**x, **y})
        self.table = pd.concat([self.table, new_table]) if append else new_table
        self.table.index = np.arange(len(self.table))

        self._update_models(train=train_models, a_priori_hypers=hypers)

    def _update_models(self, train=True, skew_dims=None, a_priori_hypers=None):
        skew_dims = skew_dims if skew_dims is not None else self.latent_dim_tuples

        # if self.initialized:
        #     cached_hypers = self.hypers

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
                    torch.tensor(1e-2).square(),
                    torch.tensor(1 / obj.min_snr).square(),
                ),
                # noise_prior=gpytorch.priors.torch_priors.LogNormalPrior(loc=loc, scale=scale),
            )

            outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)  # , batch_shape=torch.Size((1,)))

            obj.model = models.LatentGP(
                train_inputs=train_inputs,
                train_targets=train_targets,
                likelihood=likelihood,
                skew_dims=skew_dims,
                input_transform=self._subset_input_transform(active=True),
                outcome_transform=outcome_transform,
            )

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            self.all_objectives_valid.long(), learn_additional_noise=True
        )

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=torch.tensor(inputs).double(),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            skew_dims=skew_dims,
            likelihood=dirichlet_likelihood,
            input_transform=self._subset_input_transform(active=True),
        )

        if a_priori_hypers is not None:
            self._set_hypers(a_priori_hypers)
        else:
            self._train_models()
            # try:

            # except botorch.exceptions.errors.ModelFittingError:
            #     if self.initialized:
            #         self._set_hypers(cached_hypers)
            #     else:
            #         raise RuntimeError("Could not fit model on initialization!")

        self.constraint = GenericDeterministicModel(f=lambda x: self.classifier.probabilities(x)[..., -1])

    def ask(self, acq_func_identifier="qei", n=1, route=True, sequential=True):
        """Ask the agent for the best point to sample, given an acquisition function.

        Parameters
        ----------
        acq_func_identifier :
            Which acquisition function to use. Supported values can be found in `agent.all_acq_funcs`
        n : int
            How many points you want
        route : bool
            Whether to route the supplied points to make a more efficient path.
        sequential : bool
            Whether to generate points sequentially (as opposed to in parallel). Sequential generation involves
            finding one points and constructing a fantasy posterior about its value to generate the next point.
        """

        acq_func_name = acquisition.parse_acq_func_identifier(acq_func_identifier)
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

            acq_func, acq_func_meta = self.get_acquisition_function(identifier=acq_func_identifier, return_metadata=True)

            NUM_RESTARTS = 8
            RAW_SAMPLES = 1024

            candidates, acqf_obj = botorch.optim.optimize_acqf(
                acq_function=acq_func,
                bounds=self.acquisition_function_bounds,
                q=n,
                sequential=sequential,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            )

            x = candidates.numpy().astype(float)

            active_dofs_are_read_only = np.array([dof.read_only for dof in self.dofs.subset(active=True)])

            acq_points = x[..., ~active_dofs_are_read_only]
            read_only_X = x[..., active_dofs_are_read_only]
            acq_func_meta["read_only_values"] = read_only_X

        else:
            acqf_obj = None

            if acq_func_name == "random":
                acq_points = torch.rand()
                acq_func_meta = {"name": "random", "args": {}}

            if acq_func_name == "quasi-random":
                acq_points = self._subset_inputs_sampler(n=n, active=True, read_only=False).squeeze(1).numpy()
                acq_func_meta = {"name": "quasi-random", "args": {}}

            elif acq_func_name == "grid":
                n_active_dims = len(self.dofs.subset(active=True, read_only=False))
                acq_points = self.test_inputs_grid(max_inputs=n).reshape(-1, n_active_dims).numpy()
                acq_func_meta = {"name": "grid", "args": {}}

            else:
                raise ValueError()

            # define dummy acqf objective
            acqf_obj = None

        acq_func_meta["duration"] = duration = ttime.monotonic() - start_time

        if self.verbose:
            print(
                f"found points {acq_points} with acqf {acq_func_meta['name']} in {duration:.01f} seconds (obj = {acqf_obj})"
            )

        if route and n > 1:
            routing_index = utils.route(self.dofs.subset(active=True, read_only=False).readback, acq_points)
            acq_points = acq_points[routing_index]

        return acq_points, acq_func_meta

    def acquire(self, acquisition_inputs):
        """Acquire and digest according to the self's acquisition and digestion plans.

        Parameters
        ----------
        acquisition_inputs :
            A 2D numpy array comprising inputs for the active and non-read-only DOFs to sample.
        """

        if self.db is None:
            raise ValueError("Cannot run acquistion without databroker instance!")

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

        except KeyboardInterrupt as interrupt:
            raise interrupt

        except Exception as error:
            if not self.tolerate_acquisition_errors:
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
        data_file=None,
        hypers_file=None,
        append=True,
    ):
        """This returns a Bluesky plan which iterates the learning algorithm, looping over ask -> acquire -> tell.

        For example:

        RE(agent.learn("qr", n=16))
        RE(agent.learn("qei", n=4, iterations=4))

        Parameters
        ----------
        acq_func : str
            A valid identifier for an implemented acquisition function.
        n : int
            How many points to sample on each iteration.
        iterations: int
            How many iterations of the learning loop to perform.
        train: bool
            Whether to train the models upon telling the agent.
        append: bool
            If `True`, add the new data to the old data. If `False`, replace the old data with the new data.
        data_file: str
            If supplied, read a saved data file instead of running the acquisition plan.
        hypers_file: str
            If supplied, read a saved hyperparameter file instead of fitting models. NOTE: The agent will assume these
            hyperparameters a priori for the rest of the run, and not try to fit a model.
        """

        if data_file is not None:
            new_table = pd.read_hdf(data_file, key="table")

        elif acq_func is not None:
            if self.sample_center_on_init and not self.initialized:
                center_inputs = np.atleast_2d(self.dofs.subset(active=True, read_only=False).limits.mean(axis=1))
                new_table = yield from self.acquire(center_inputs)
                new_table.loc[:, "acq_func"] = "sample_center_on_init"

            for i in range(iterations):
                print(f"running iteration {i + 1} / {iterations}")
                for single_acq_func in np.atleast_1d(acq_func):
                    x, acq_func_meta = self.ask(n=n, acq_func_identifier=single_acq_func)
                    new_table = yield from self.acquire(x)
                    new_table.loc[:, "acq_func"] = acq_func_meta["name"]

        else:
            raise ValueError("You must supply either an acquisition function or a filepath!")

        x = {key: new_table.pop(key).tolist() for key in self.dofs.names}
        y = {key: new_table.pop(key).tolist() for key in self.objectives.keys}
        metadata = new_table.to_dict(orient="list")

        self.tell(x=x, y=y, metadata=metadata, append=append, train_models=True)

        self.initialized = True

    def get_acquisition_function(self, identifier, return_metadata=False):
        """Returns a BoTorch acquisition function for a given identifier. Acquisition functions can be
        found in `agent.all_acq_funcs`.
        """
        return acquisition.get_acquisition_function(self, identifier=identifier, return_metadata=return_metadata)

    def reset(self):
        """Reset the agent."""
        self.table = pd.DataFrame()
        self.initialized = False

    def benchmark(
        self, output_dir="./", runs=16, n_init=64, learning_kwargs_list=[{"acq_func": "qei", "n": 4, "iterations": 16}]
    ):
        """Iterate over having the agent learn from scratch, and save the results to an output directory.

        Parameters
        ----------
        output_dir :
            Where to save the optimized agents
        runs : int
            How many benchmarks to run
        n_init : int
            How many points to sample on reseting the agent.
        learning_kwargs_list:
            A list of kwargs which the agent will run sequentially for each run.
        """
        # cache_limits = {dof.name: dof.limits for dof in self.dofs}

        for run in range(runs):
            # for dof in self.dofs:
            #     offset = 0.25 * np.ptp(dof.limits) * np.random.uniform(low=-1, high=1)
            #     dof.limits = (cache_limits[dof.name][0] + offset, cache_limits[dof.name][1] + offset)

            self.reset()

            yield from self.learn("qr", n=n_init)

            for kwargs in learning_kwargs_list:
                yield from self.learn(**kwargs)

            self.save_data(output_dir + f"benchmark-{int(ttime.time())}.h5")

    @property
    def model(self):
        """A model encompassing all the objectives. A single GP in the single-objective case, or a model list."""
        return ModelListGP(*[obj.model for obj in self.objectives]) if len(self.objectives) > 1 else self.objectives[0].model

    @property
    def objective_weights_torch(self):
        return torch.tensor(self.objectives.weights, dtype=torch.double)

    def _get_objective_targets(self, i):
        """Returns the targets (what we fit to) for an objective, given the objective index."""
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
        """Returns a (num_objectives x n_observations) array of objectives"""
        return len(self.objectives)

    @property
    def objectives_targets(self):
        """Returns a (num_objectives x n_obs) array of objectives"""
        return torch.cat([torch.tensor(self._get_objective_targets(i))[..., None] for i in range(self.n_objs)], dim=1)

    @property
    def scalarized_objectives(self):
        """Returns a (n_obs,) array of scalarized objectives"""
        return (self.objectives_targets * self.objectives.weights).sum(axis=-1)

    @property
    def max_scalarized_objective(self):
        """Returns the value of the best scalarized objective seen so far."""
        f = self.scalarized_objectives
        return np.max(np.where(np.isnan(f), -np.inf, f))

    @property
    def argmax_scalarized_objective(self):
        """Returns the index of the best scalarized objective seen so far."""
        f = self.scalarized_objectives
        return np.argmax(np.where(np.isnan(f), -np.inf, f))

    @property
    def all_objectives_valid(self):
        """A mask of whether all objectives are valid for each data point."""
        return ~torch.isnan(self.scalarized_objectives)

    def test_inputs_grid(self, max_inputs=MAX_TEST_INPUTS):
        """Returns a (`n_side`, ..., `n_side`, 1, `n_active_dofs`) grid of test_inputs; `n_side` is 1 if a dof is read-only.
        The value of `n_side` is the largest value such that the entire grid has less than `max_inputs` inputs.
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
        """Returns a (n, 1, n_active_dof) grid of test_inputs"""
        return utils.sobol_sampler(self.acquisition_function_bounds, n=n)

    @property
    def acquisition_function_bounds(self):
        """Returns a (2, n_active_dof) array of bounds for the acquisition function"""
        active_dofs = self.dofs.subset(active=True)

        acq_func_lower_bounds = [dof.lower_limit if not dof.read_only else dof.readback for dof in active_dofs]
        acq_func_upper_bounds = [dof.upper_limit if not dof.read_only else dof.readback for dof in active_dofs]

        return torch.tensor(np.vstack([acq_func_lower_bounds, acq_func_upper_bounds]), dtype=torch.double)

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

    def forget(self, index, train=True):
        """
        Make the agent forget some index of the data table.
        """
        self.table.drop(index=index, inplace=True)
        self._update_models(train=train)

    def forget_last_n(self, n, train=True):
        """
        Make the agent forget the last `n` data points taken.
        """
        if n > len(self.table):
            raise ValueError(f"Cannot forget {n} data points (only {len(self.table)} have been taken).")
        self.forget(self.table.index.iloc[-n:], train=train)

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
        """Returns a dict of all the hyperparameters in all the agent's models."""
        hypers = {"classifier": {}}
        for key, value in self.classifier.state_dict().items():
            hypers["classifier"][key] = value
        for obj in self.objectives:
            hypers[obj.key] = {}
            for key, value in obj.model.state_dict().items():
                hypers[obj.key][key] = value

        return hypers

    def save_hypers(self, filepath):
        """Save the agent's fitted hyperparameters to a given filepath."""
        hypers = self.hypers
        with h5py.File(filepath, "w") as f:
            for model_key in hypers.keys():
                f.create_group(model_key)
                for param_key, param_value in hypers[model_key].items():
                    f[model_key].create_dataset(param_key, data=param_value)

    @staticmethod
    def load_hypers(filepath):
        """Load hyperparameters from a file."""
        hypers = {}
        with h5py.File(filepath, "r") as f:
            for model_key in f.keys():
                hypers[model_key] = OrderedDict()
                for param_key, param_value in f[model_key].items():
                    hypers[model_key][param_key] = torch.tensor(np.atleast_1d(param_value[()]))
        return hypers

    def _train_models(self, **kwargs):
        """Fit all of the agent's models. All kwargs are passed to `botorch.fit.fit_gpytorch_mll`."""
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
    def all_acq_funcs(self):
        """Description and identifiers for all supported acquisition functions."""
        entries = []
        for k, d in acquisition.config.items():
            ret = ""
            ret += f'{d["pretty_name"].upper()} (identifiers: {d["identifiers"]})\n'
            ret += f'-> {d["description"]}'
            entries.append(ret)

        print("\n\n".join(entries))

    @property
    def inputs(self):
        """A two-dimensional array of all DOF values."""
        return self.table.loc[:, self.dofs.names].astype(float)

    @property
    def active_inputs(self):
        """A two-dimensional array of all inputs for model fitting."""
        return self.table.loc[:, self.dofs.subset(active=True).names].astype(float)

    @property
    def acquisition_inputs(self):
        """A two-dimensional array of all inputs for computing acquisition functions."""
        return self.table.loc[:, self.dofs.subset(active=True, read_only=False).names].astype(float)

    @property
    def best(self):
        """Returns all data for the best point."""
        return self.table.loc[self.argmax_scalarized_objective]

    @property
    def best_inputs(self):
        """Returns the value of each DOF at the best point."""
        return self.table.loc[self.argmax_scalarized_objective, self.dofs.names].to_dict()

    def go_to(self, **positions):
        """Set all settable DOFs to a given position. DOF/value pairs should be supplied as kwargs, e.g. as

        RE(agent.go_to(some_dof=x1, some_other_dof=x2, ...))
        """
        mv_args = []
        for dof_name, dof_value in positions.items():
            if dof_name not in self.dofs.names:
                raise ValueError(f"There is no DOF named {dof_name}")
            dof = self.dofs[dof_name]
            if dof.read_only:
                raise ValueError(f"Cannot move DOF {dof_name} as it is read-only.")
            mv_args.append(dof.device)
            mv_args.append(dof_value)

        yield from bps.mv(*mv_args)

    def go_to_best(self):
        """Go to the position of the best input seen so far."""
        yield from self.go_to(**self.best_inputs)

    def plot_objectives(self, axes: Tuple = (0, 1), **kwargs):
        """Plot the sampled objectives

        Parameters
        ----------
        axes :
            A tuple specifying which DOFs to plot as a function of. Can be either an int or the name of DOFs.
        """
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_objs_one_dof(self, **kwargs)
        else:
            plotting._plot_objs_many_dofs(self, axes=axes, **kwargs)

    def plot_acquisition(self, acq_func="ei", axes: Tuple = (0, 1), **kwargs):
        """Plot an acquisition function over test inputs sampling the limits of the parameter space.

        Parameters
        ----------
        acq_func :
            Which acquisition function to plot. Can also take a list of acquisition functions.
        axes :
            A tuple specifying which DOFs to plot as a function of. Can be either an int or the name of DOFs.
        """
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_acqf_one_dof(self, acq_funcs=np.atleast_1d(acq_func), **kwargs)

        else:
            plotting._plot_acqf_many_dofs(self, acq_funcs=np.atleast_1d(acq_func), axes=axes, **kwargs)

    def plot_constraint(self, axes: Tuple = (0, 1), **kwargs):
        """Plot the modeled constraint over test inputs sampling the limits of the parameter space.

        Parameters
        ----------
        axes :
            A tuple specifying which DOFs to plot as a function of. Can be either an int or the name of DOFs.
        """
        if len(self.dofs.subset(active=True, read_only=False)) == 1:
            plotting._plot_valid_one_dof(self, **kwargs)

        else:
            plotting._plot_valid_many_dofs(self, axes=axes, **kwargs)

    def plot_history(self, **kwargs):
        """Plot the improvement of the agent over time."""
        plotting._plot_history(self, **kwargs)
