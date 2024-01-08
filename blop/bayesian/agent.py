import logging
import time as ttime
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Callable, Optional, Sequence, Tuple

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import botorch
import gpytorch
import h5py
import matplotlib as mpl
import napari
import numpy as np
import pandas as pd
import scipy as sp
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from databroker import Broker
from ophyd import Signal

from .. import utils
from . import acquisition, models, plotting
from .digestion import default_digestion_function
from .dofs import DOF, DOFList
from .objectives import Objective, ObjectiveList
from .plans import default_acquisition_plan
from .transforms import TargetingPosteriorTransform

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
        train_every: int = 1,
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
        # 'read': the agent will read the input on every acquisition (all dofs are always read)
        # 'move': the agent will try to set and optimize over these (there must be at least one of these)
        # 'input' means that the agent will use the value to make its posterior
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

        self.train_every = train_every
        self.trigger_delay = trigger_delay
        self.sample_center_on_init = sample_center_on_init

        self.table = pd.DataFrame()

        self.initialized = False
        self.a_priori_hypers = None

        self.n_last_trained = 0

    def view(self, item: str = "mean", cmap: str = "turbo", max_inputs: int = MAX_TEST_INPUTS):
        """
        Use napari to see a high-dimensional array.

        Parameters
        ----------
        item : str
            The thing to be viewed. Either 'mean', 'error', or an acquisition function.
        """

        test_grid = self.test_inputs_grid(max_inputs=max_inputs)

        self.viewer = napari.Viewer()

        if item in ["mean", "error"]:
            for obj in self.objectives:
                p = obj.model.posterior(test_grid)

                if item == "mean":
                    mean = p.mean.detach().numpy()[..., 0, 0]
                    self.viewer.add_image(data=mean, name=f"{obj.name}_mean", colormap=cmap)

                if item == "error":
                    error = np.sqrt(p.variance.detach().numpy()[..., 0, 0])
                    self.viewer.add_image(data=error, name=f"{obj.name}_error", colormap=cmap)

        else:
            try:
                acq_func_identifier = acquisition.parse_acq_func_identifier(identifier=item)
            except Exception:
                raise ValueError("'item' must be either 'mean', 'error', or a valid acq func.")

            acq_func, acq_func_meta = self.get_acquisition_function(identifier=acq_func_identifier, return_metadata=True)
            a = acq_func(test_grid).detach().numpy()

            self.viewer.add_image(data=a, name=f"{acq_func_identifier}", colormap=cmap)

        self.viewer.dims.axis_labels = self.dofs.names

    def tell(
        self,
        data: Optional[Mapping] = {},
        x: Optional[Mapping] = {},
        y: Optional[Mapping] = {},
        metadata: Optional[Mapping] = {},
        append=True,
        train=True,
        hypers=None,
    ):
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
        _train_models: bool
            Whether to train the models on construction.
        hypers:
            A dict of hyperparameters for the model to assume a priori, instead of training.
        """

        if not data:
            if not x and y:
                raise ValueError("Must supply either x and y, or data.")
            data = {**x, **y, **metadata}

        data = {k: np.atleast_1d(v) for k, v in data.items()}
        unique_field_lengths = {len(v) for v in data.values()}

        if len(unique_field_lengths) > 1:
            raise ValueError("All supplies values must be the same length!")

        new_table = pd.DataFrame(data)
        self.table = pd.concat([self.table, new_table]) if append else new_table
        self.table.index = np.arange(len(self.table))

        for obj in self.objectives:
            t0 = ttime.monotonic()

            cached_hypers = obj.model.state_dict() if hasattr(obj, "model") else None

            obj.model = self._construct_model(obj)

            if len(obj.model.train_targets) >= 2:
                t0 = ttime.monotonic()
                self._train_model(obj.model, hypers=(None if train else cached_hypers))
                if self.verbose:
                    print(f"trained model '{obj.name}' in {1e3*(ttime.monotonic() - t0):.00f} ms")

        # TODO: should this be per objective?
        self._construct_classifier()

    def ask(self, acq_func_identifier="qei", n=1, route=True, sequential=True, upsample=1, **acq_func_kwargs):
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
            print(f"finding points with acquisition function '{acq_func_name}' ...")

        if acq_func_type in ["analytic", "monte_carlo"]:
            if not all(hasattr(obj, "model") for obj in self.objectives):
                raise RuntimeError(
                    f"Can't construct non-trivial acquisition function '{acq_func_identifier}'"
                    f" (the agent is not initialized!)"
                )

            if acq_func_type == "analytic" and n > 1:
                raise ValueError("Can't generate multiple design points for analytic acquisition functions.")

            acq_func, acq_func_meta = self.get_acquisition_function(
                identifier=acq_func_identifier, return_metadata=True, **acq_func_kwargs
            )

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

            # this includes both RO and non-RO DOFs
            candidates = candidates.numpy()

            active_dofs_are_read_only = np.array([dof.read_only for dof in self.dofs.subset(active=True)])

            acq_points = candidates[..., ~active_dofs_are_read_only]
            read_only_values = candidates[..., active_dofs_are_read_only]
            acq_func_meta["read_only_values"] = read_only_values

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
            acqf_obj = 0

        acq_func_meta["duration"] = duration = ttime.monotonic() - start_time

        if self.verbose:
            print(f"found points {acq_points} in {1e3*duration:.01f} ms (obj = {acqf_obj})")

        if route and n > 1:
            routing_index = utils.route(self.dofs.subset(active=True, read_only=False).readback, acq_points)
            acq_points = acq_points[routing_index]

        if upsample > 1:
            idx = np.arange(len(acq_points))
            upsampled_idx = np.linspace(0, len(idx) - 1, upsample * len(idx) - 1)
            acq_points = sp.interpolate.interp1d(idx, acq_points, axis=0)(upsampled_idx)

        res = {
            "points": acq_points,
            "acq_func": acq_func_meta["name"],
            "acq_func_kwargs": acq_func_kwargs,
            "duration": acq_func_meta["duration"],
            "sequential": sequential,
            "upsample": upsample,
            "read_only_values": acq_func_meta.get("read_only_values"),
        }

        return res

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
            #         products.loc[index, objective['key']] = getattr(entry, objective['key'])

        except KeyboardInterrupt as interrupt:
            raise interrupt

        except Exception as error:
            if not self.tolerate_acquisition_errors:
                raise error
            logging.warning(f"Error in acquisition/digestion: {repr(error)}")
            products = pd.DataFrame(acquisition_inputs, columns=self.dofs.subset(active=True, read_only=False).names)
            for obj in self.objectives:
                products.loc[:, obj.name] = np.nan

        if not len(acquisition_inputs) == len(products):
            raise ValueError("The table returned by the digestion function must be the same length as the sampled inputs!")

        return products

    def load_data(self, data_file, append=True, train=True):
        new_table = pd.read_hdf(data_file, key="table")
        x = {key: new_table.pop(key).tolist() for key in self.dofs.names}
        y = {key: new_table.pop(key).tolist() for key in self.objectives.names}
        metadata = new_table.to_dict(orient="list")
        self.tell(x=x, y=y, metadata=metadata, append=append, train=train)

    def learn(
        self,
        acq_func: str = "qei",
        n: int = 1,
        iterations: int = 1,
        upsample: int = 1,
        train: bool = True,
        append: bool = True,
        hypers_file=None,
    ):
        """This returns a Bluesky plan which iterates the learning algorithm, looping over ask -> acquire -> tell.

        For example:

        RE(agent.learn('qr', n=16))
        RE(agent.learn('qei', n=4, iterations=4))

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

        if self.sample_center_on_init and not self.initialized:
            center_inputs = np.atleast_2d(self.dofs.subset(active=True, read_only=False).limits.mean(axis=1))
            new_table = yield from self.acquire(center_inputs)
            new_table.loc[:, "acq_func"] = "sample_center_on_init"

        for i in range(iterations):
            print(f"running iteration {i + 1} / {iterations}")
            for single_acq_func in np.atleast_1d(acq_func):
                res = self.ask(n=n, acq_func_identifier=single_acq_func, upsample=upsample)
                new_table = yield from self.acquire(res["points"])
                new_table.loc[:, "acq_func"] = res["acq_func"]

                x = {key: new_table.pop(key).tolist() for key in self.dofs.names}
                y = {key: new_table.pop(key).tolist() for key in self.objectives.names}
                metadata = new_table.to_dict(orient="list")
                self.tell(x=x, y=y, metadata=metadata, append=append, train=train)

    def _train_model(self, model, hypers=None, **kwargs):
        """Fit all of the agent's models. All kwargs are passed to `botorch.fit.fit_gpytorch_mll`."""
        if hypers is not None:
            model.load_state_dict(hypers)
        else:
            botorch.fit.fit_gpytorch_mll(gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model), **kwargs)
        model.trained = True

    def _construct_model(self, obj, skew_dims=None):
        """
        Construct an untrained model for an objective.
        """

        skew_dims = skew_dims if skew_dims is not None else self.latent_dim_tuples

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(
                torch.tensor(1e-4).square(),
                torch.tensor(1 / obj.min_snr).square(),
            ),
            # noise_prior=gpytorch.priors.torch_priors.LogNormalPrior(loc=loc, scale=scale),
        )

        outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)  # , batch_shape=torch.Size((1,)))

        train_inputs = self.train_inputs(active=True)
        train_targets = self.train_targets(obj.name)

        safe = ~(torch.isnan(train_inputs).any(axis=1) | torch.isnan(train_targets).any(axis=1))

        model = models.LatentGP(
            train_inputs=train_inputs[safe],
            train_targets=train_targets[safe],
            likelihood=likelihood,
            skew_dims=skew_dims,
            input_transform=self.input_transform,
            outcome_transform=outcome_transform,
        )

        model.trained = False

        return model

    def _construct_classifier(self, skew_dims=None):
        skew_dims = skew_dims if skew_dims is not None else self.latent_dim_tuples

        dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            self.all_objectives_valid.long(), learn_additional_noise=True
        )

        self.classifier = models.LatentDirichletClassifier(
            train_inputs=self.train_inputs(active=True),
            train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2).double(),
            skew_dims=skew_dims,
            likelihood=dirichlet_likelihood,
            input_transform=self.input_transform,
        )

        self._train_model(self.classifier)
        self.constraint = GenericDeterministicModel(f=lambda x: self.classifier.probabilities(x)[..., -1])

    def get_acquisition_function(self, identifier, return_metadata=False):
        """Returns a BoTorch acquisition function for a given identifier. Acquisition functions can be
        found in `agent.all_acq_funcs`.
        """
        return acquisition.get_acquisition_function(self, identifier=identifier, return_metadata=return_metadata)

    def reset(self):
        """Reset the agent."""
        self.table = pd.DataFrame()

        for obj in self.objectives:
            if hasattr(obj, "model"):
                del obj.model

        self.n_last_trained = 0

    def benchmark(
        self, output_dir="./", runs=16, n_init=64, learning_kwargs_list=[{"acq_func": "qei", "n": 4, "iterations": 16}]
    ):
        """Iterate over having the agent learn from scratch, and save the results to an output directory.

        Parameters
        ----------
        output_dir :
            Where to save the agent output.
        runs : int
            How many benchmarks to run
        learning_kwargs_list:
            A list of kwargs to pass to the learn method which the agent will run sequentially for each run.
        """

        for run in range(runs):
            self.reset()

            for kwargs in learning_kwargs_list:
                yield from self.learn(**kwargs)

            self.save_data(output_dir + f"benchmark-{int(ttime.time())}.h5")

    @property
    def model(self):
        """A model encompassing all the objectives. A single GP in the single-objective case, or a model list."""
        return ModelListGP(*[obj.model for obj in self.objectives]) if len(self.objectives) > 1 else self.objectives[0].model

    def posterior(self, x):
        """A model encompassing all the objectives. A single GP in the single-objective case, or a model list."""
        return self.model.posterior(x)

    @property
    def objective_weights_torch(self):
        return torch.tensor(self.objectives.weights, dtype=torch.double)

    @property
    def scalarizing_transform(self):
        return ScalarizedPosteriorTransform(weights=self.objective_weights_torch, offset=0)

    @property
    def targeting_transform(self):
        return TargetingPosteriorTransform(weights=self.objective_weights_torch, targets=self.objectives.targets)

    @property
    def pseudo_targets(self):
        """Targets for the posterior transform"""
        return torch.tensor(
            [
                self.train_targets(active=True)[..., i].max()
                if t == "max"
                else self.train_targets(active=True)[..., i].min()
                if t == "min"
                else t
                for i, t in enumerate(self.objectives.targets)
            ]
        )

    @property
    def scalarized_objectives(self):
        """Returns a (n_obs,) array of scalarized objectives"""
        return self.targeting_transform.evaluate(self.train_targets(active=True)).sum(axis=-1)

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
        return (
            torch.cat(
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
            )
            .unsqueeze(-2)
            .double()
        )

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

    @property
    def input_transform(self):
        """
        A bounding transform for all the active DOFs. This is used for model fitting.
        """
        return self._subset_input_transform(active=True)

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
        self.__construct_models(train=train)

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
            obj.model.load_state_dict(hypers[obj.name])
        self.classifier.load_state_dict(hypers["classifier"])

    @property
    def hypers(self):
        """Returns a dict of all the hyperparameters in all the agent's models."""
        hypers = {"classifier": {}}
        for key, value in self.classifier.state_dict().items():
            hypers["classifier"][key] = value
        for obj in self.objectives:
            hypers[obj.name] = {}
            for key, value in obj.model.state_dict().items():
                hypers[obj.name][key] = value

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

    def __train_models(self, **kwargs):
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

        self.n_last_trained = len(self.table)

    @property
    def all_acq_funcs(self):
        """Description and identifiers for all supported acquisition functions."""
        entries = []
        for k, d in acquisition.config.items():
            ret = ""
            ret += f"{d['pretty_name'].upper()} (identifiers: {d['identifiers']})\n"
            ret += f"-> {d['description']}"
            entries.append(ret)

        print("\n\n".join(entries))

    @property
    def inputs(self):
        """A DataFrame of all DOF values."""
        return self.table.loc[:, self.dofs.names].astype(float)

    def train_inputs(self, dof_name=None, **subset_kwargs):
        """A two-dimensional tensor of all DOF values."""

        if dof_name is None:
            return torch.cat([self.train_inputs(dof.name) for dof in self.dofs.subset(**subset_kwargs)], dim=-1)

        dof = self.dofs[dof_name]
        inputs = self.table.loc[:, dof.name].values.copy()

        # check that inputs values are inside acceptable values
        valid = (inputs >= dof.limits[0]) & (inputs <= dof.limits[1])
        inputs = np.where(valid, inputs, np.nan)

        # transform if needed
        if dof.log:
            inputs = np.where(inputs > 0, np.log(inputs), np.nan)

        return torch.tensor(inputs, dtype=torch.double).unsqueeze(-1)

    def train_targets(self, obj_name=None, **subset_kwargs):
        """Returns the values associated with an objective name."""

        if obj_name is None:
            return torch.cat([self.train_targets(obj.name) for obj in self.objectives], dim=-1)

        obj = self.objectives[obj_name]
        targets = self.table.loc[:, obj.name].values.copy()

        # check that targets values are inside acceptable values
        valid = (targets >= obj.limits[0]) & (targets <= obj.limits[1])
        targets = np.where(valid, targets, np.nan)

        # transform if needed
        if obj.log:
            targets = np.where(targets > 0, np.log(targets), np.nan)

        return torch.tensor(targets, dtype=torch.double).unsqueeze(-1)

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
