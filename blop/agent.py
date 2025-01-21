import logging
import os
import pathlib
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
import numpy as np
import pandas as pd
import scipy as sp
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from databroker import Broker
from ophyd import Signal

from . import plotting, utils
from .bayesian import acquisition, models
from .bayesian.acquisition import _construct_acqf, parse_acqf_identifier
from .bayesian.models import construct_single_task_model, train_model

# from .bayesian.transforms import TargetingPosteriorTransform
from .digestion import default_digestion_function
from .dofs import DOF, DOFList
from .objectives import Objective, ObjectiveList
from .plans import default_acquisition_plan

logger = logging.getLogger("maria")

warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

mpl.rc("image", cmap="coolwarm")

DEFAULT_MAX_SAMPLES = 3200


def _validate_dofs_and_objs(dofs: DOFList, objs: ObjectiveList):
    if len(dofs) == 0:
        raise ValueError("You must supply at least one DOF.")

    if len(objs) == 0:
        raise ValueError("You must supply at least one objective.")

    for obj in objs:
        for latent_group in obj.latent_groups:
            for dof_name in latent_group:
                if dof_name not in dofs.names:
                    warnings.warn(
                        f"DOF name '{dof_name}' in latent group for objective '{obj.name}' does not exist."
                        "it will be ignored."
                    )


class Agent:
    def __init__(
        self,
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        db: Broker = None,
        detectors: Sequence[Signal] = None,
        acquistion_plan=default_acquisition_plan,
        digestion: Callable = default_digestion_function,
        digestion_kwargs: dict = {},
        verbose: bool = False,
        enforce_all_objectives_valid: bool = True,
        exclude_pruned: bool = True,
        model_inactive_objectives: bool = False,
        tolerate_acquisition_errors: bool = False,
        sample_center_on_init: bool = False,
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
        detectors : iterable of ophyd objects
            Detectors to trigger during acquisition.
        acquisition_plan : optional
            A plan that samples the beamline for some given inputs.
        digestion :
            A function to digest the output of the acquisition, taking a DataFrame as an argument.
        digestion_kwargs :
            Some kwargs for the digestion function.
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
        self.detectors = list(np.atleast_1d(detectors or []))

        _validate_dofs_and_objs(self.dofs, self.objectives)

        self.db = db
        self.acquisition_plan = acquistion_plan
        self.digestion = digestion
        self.digestion_kwargs = digestion_kwargs

        self.verbose = verbose

        self.model_inactive_objectives = model_inactive_objectives
        self.tolerate_acquisition_errors = tolerate_acquisition_errors
        self.enforce_all_objectives_valid = enforce_all_objectives_valid
        self.exclude_pruned = exclude_pruned

        self.train_every = train_every
        self.trigger_delay = trigger_delay
        self.sample_center_on_init = sample_center_on_init

        self._table = pd.DataFrame()

        self.initialized = False
        self.a_priori_hypers = None

        self.n_last_trained = 0

    @property
    def table(self):
        return self._table

    def unpack_run(self):
        return

    def measurement_plan(self):
        return

    def __iter__(self):
        for index in range(len(self)):
            yield self.dofs[index]

    def __getattr__(self, attr):
        acqf_config = acquisition.parse_acqf_identifier(attr, strict=False)
        if acqf_config is not None:
            acqf, _ = _construct_acqf(agent=self, acqf_name=acqf_config["name"])
            return acqf
        raise AttributeError(f"No attribute named '{attr}'.")

    def refresh(self):
        self._construct_all_models()
        self._train_all_models()

    def redigest(self):
        self._table = self.digestion(self._table, **self.digestion_kwargs)

    def sample(self, n: int = DEFAULT_MAX_SAMPLES, normalize: bool = False, method: str = "quasi-random") -> torch.Tensor:
        """
        Returns a (..., 1, n_active_dofs) tensor of points sampled within the parameter space.

        Parameters
        ----------
        n : int
            How many points to sample.
        method : str
            How to sample the points. Must be one of 'quasi-random', 'random', or 'grid'.
        normalize: bool
            If True, sample the unit hypercube. If False, sample the parameter space of the agent.
        """

        active_dofs = self.dofs(active=True)

        if method == "quasi-random":
            X = utils.normalized_sobol_sampler(n, d=len(active_dofs))

        elif method == "random":
            X = torch.rand(size=(n, 1, len(active_dofs)))

        elif method == "grid":
            n_side_if_settable = int(np.power(n, 1 / np.sum(~active_dofs.read_only)))
            sides = [
                torch.linspace(0, 1, n_side_if_settable) if not dof.read_only else torch.zeros(1) for dof in active_dofs
            ]
            X = torch.cat([x.unsqueeze(-1) for x in torch.meshgrid(sides, indexing="ij")], dim=-1).unsqueeze(-2).double()

        else:
            raise ValueError("'method' argument must be one of ['quasi-random', 'random', 'grid'].")

        return X.double() if normalize else self.dofs(active=True).untransform(X).double()

    def ask(self, acqf="qei", n=1, route=True, sequential=True, upsample=1, **acqf_kwargs):
        """Ask the agent for the best point to sample, given an acquisition function.

        Parameters
        ----------
        acqf_identifier :
            Which acquisition function to use. Supported values can be found in `agent.all_acqfs`
        n : int
            How many points you want
        route : bool
            Whether to route the supplied points to make a more efficient path.
        sequential : bool
            Whether to generate points sequentially (as opposed to in parallel). Sequential generation involves
            finding one points and constructing a fantasy posterior about its value to generate the next point.
        """

        acqf_config = parse_acqf_identifier(acqf)
        if acqf_config is None:
            raise ValueError(f"'{acqf}' is an invalid acquisition function.")

        start_time = ttime.monotonic()

        active_dofs = self.dofs(active=True)
        active_objs = self.objectives(active=True)

        # these are the fake acquisiton functions that we don't need to construct
        if acqf_config["name"] in ["quasi-random", "random", "grid"]:
            candidates = self.sample(n=n, method=acqf_config["name"]).squeeze(1).numpy()

            # define dummy acqf kwargs and objective
            acqf_kwargs, acqf_obj = {}, torch.zeros(len(candidates))

        else:
            # check that all the objectives have models
            if not all(hasattr(obj, "_model") for obj in active_objs):
                raise RuntimeError(
                    f"Can't construct non-trivial acquisition function '{acqf}' as the agent is not initialized."
                )

            # if the model for any active objective mismatches the active dofs, reconstrut and train it
            for obj in active_objs:
                if obj.model_dofs != set(active_dofs.names):
                    self._construct_model(obj)
                    train_model(obj.model)

            if acqf_config["type"] == "analytic" and n > 1:
                raise ValueError("Can't generate multiple design points for analytic acquisition functions.")

            # we may pick up some more kwargs
            acqf, acqf_kwargs = _construct_acqf(self, acqf_name=acqf_config["name"], **acqf_kwargs)

            NUM_RESTARTS = 8
            RAW_SAMPLES = 256

            candidates, acqf_obj = botorch.optim.optimize_acqf(
                acq_function=acqf,
                bounds=self.sample_domain,
                q=n,
                sequential=sequential,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                fixed_features={i: dof._transform(dof.readback) for i, dof in enumerate(active_dofs) if dof.read_only},
            )

            # this includes both RO and non-RO DOFs.
            # and is in the transformed model space
            candidates = self.dofs(active=True).untransform(candidates).numpy()

        # p = self.posterior(candidates) if hasattr(self, "model") else None

        active_dofs = self.dofs(active=True)

        points = candidates[..., ~active_dofs.read_only]
        read_only_values = candidates[..., active_dofs.read_only]

        duration = 1e3 * (ttime.monotonic() - start_time)

        if route and n > 1:
            current_points = np.array([dof.readback for dof in active_dofs if not dof.read_only])
            travel_expenses = np.array([dof.travel_expense for dof in active_dofs if not dof.read_only])
            routing_index = utils.route(current_points, points, dim_weights=travel_expenses)
            points = points[routing_index]

        if upsample > 1:
            if n == 1:
                raise ValueError("Cannot upsample points unless n > 1.")
            idx = np.arange(len(points))
            upsampled_idx = np.linspace(0, len(idx) - 1, upsample * len(idx) - 1)
            points = sp.interpolate.interp1d(idx, points, axis=0)(upsampled_idx)

        res = {
            "points": {dof.name: list(points[..., i]) for i, dof in enumerate(active_dofs(read_only=False))},
            "acqf_name": acqf_config["name"],
            "acqf_obj": list(np.atleast_1d(acqf_obj.numpy())),
            "acqf_kwargs": acqf_kwargs,
            "duration_ms": duration,
            "sequential": sequential,
            "upsample": upsample,
            "read_only_values": read_only_values,
            # "posterior": p,
        }

        return res

    def tell(
        self,
        data: Optional[Mapping] = {},
        x: Optional[Mapping] = {},
        y: Optional[Mapping] = {},
        metadata: Optional[Mapping] = {},
        append: bool = True,
        update_models: bool = True,
        train: bool = None,
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
        train: bool
            Whether to train the models on construction.
        hypers:
            A dict of hyperparameters for the model to assume a priori, instead of training.
        """

        if not data:
            if not x and y:
                raise ValueError("Must supply either x and y, or data.")
            data = {**x, **y, **metadata}

        data = {k: list(np.atleast_1d(v)) for k, v in data.items()}
        unique_field_lengths = {len(v) for v in data.values()}

        if len(unique_field_lengths) > 1:
            raise ValueError("All supplies values must be the same length!")

        new_table = pd.DataFrame(data)
        self._table = pd.concat([self._table, new_table]) if append else new_table
        self._table.index = np.arange(len(self._table))

        if update_models:
            objectives_to_model = self.objectives if self.model_inactive_objectives else self.objectives(active=True)
            for obj in objectives_to_model:
                t0 = ttime.monotonic()

                cached_hypers = obj.model.state_dict() if hasattr(obj, "_model") else None
                n_before_tell = obj.n_valid
                self._construct_model(obj)
                n_after_tell = obj.n_valid

                if train is None:
                    train = int(n_after_tell / self.train_every) > int(n_before_tell / self.train_every)

                if len(obj.model.train_targets) >= 4:
                    if train:
                        t0 = ttime.monotonic()
                        train_model(obj.model)
                        if self.verbose:
                            logger.debug(f"trained model '{obj.name}' in {1e3 * (ttime.monotonic() - t0):.00f} ms")

                    else:
                        train_model(obj.model, hypers=cached_hypers)

    def learn(
        self,
        acqf: str = "qei",
        n: int = 1,
        iterations: int = 1,
        upsample: int = 1,
        train: bool = None,
        append: bool = True,
        hypers: str = None,
        route: bool = True,
        **acqf_kwargs,
    ):
        """This returns a Bluesky plan which iterates the learning algorithm, looping over ask -> acquire -> tell.

        For example:

        RE(agent.learn('qr', n=16))
        RE(agent.learn('qei', n=4, iterations=4))

        Parameters
        ----------
        acqf : str
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
            center_inputs = np.atleast_2d(self.dofs(active=True, read_only=False).search_domain.mean(axis=1))
            new_table = yield from self.acquire(center_inputs)
            new_table.loc[:, "acqf"] = "sample_center_on_init"

        for i in range(iterations):
            if self.verbose:
                logger.info(f"running iteration {i + 1} / {iterations}")
            for single_acqf in np.atleast_1d(acqf):
                res = self.ask(n=n, acqf=single_acqf, upsample=upsample, route=route, **acqf_kwargs)
                new_table = yield from self.acquire(res["points"])
                new_table.loc[:, "acqf"] = res["acqf_name"]

                x = {key: new_table.loc[:, key].tolist() for key in self.dofs.names}
                y = {key: new_table.loc[:, key].tolist() for key in self.objectives.names}
                metadata = {
                    key: new_table.loc[:, key].tolist() for key in new_table.columns if (key not in x) and (key not in y)
                }
                self.tell(x=x, y=y, metadata=metadata, append=append, train=train)

    def view(self, item: str = "mean", cmap: str = "turbo", max_inputs: int = 2**16):
        """
        Use napari to see a high-dimensional array.

        Parameters
        ----------
        item : str
            The thing to be viewed. Either 'mean', 'error', or an acquisition function.
        """

        import napari  # noqa E402

        test_grid = self.sample(n=max_inputs, method="grid")

        self.viewer = napari.Viewer()

        if item in ["mean", "error"]:
            for obj in self.objectives(active=True):
                p = obj.model.posterior(test_grid)

                if item == "mean":
                    mean = p.mean.detach().numpy()[..., 0, 0]
                    self.viewer.add_image(data=mean, name=f"{obj.name}_mean", colormap=cmap)

                if item == "error":
                    error = np.sqrt(p.variance.detach().numpy()[..., 0, 0])
                    self.viewer.add_image(data=error, name=f"{obj.name}_error", colormap=cmap)

        else:
            try:
                acqf_identifier = acquisition.parse_acqf_identifier(identifier=item)
            except Exception:
                raise ValueError("'item' must be either 'mean', 'error', or a valid acq func.")

            acqf, acqf_meta = self._get_acquisition_function(identifier=acqf_identifier, return_metadata=True)
            a = acqf(test_grid).detach().numpy()

            self.viewer.add_image(data=a, name=f"{acqf_identifier}", colormap=cmap)

        self.viewer.dims.axis_labels = self.dofs.names

    def acquire(self, points):
        """Acquire and digest according to the self's acquisition and digestion plans.

        Parameters
        ----------
        acquisition_inputs :
            A 2D numpy array comprising inputs for the active and non-read-only DOFs to sample.
        """

        if self.db is None:
            raise ValueError("Cannot run acquistion without databroker instance!")

        acquisition_dofs = self.dofs(active=True, read_only=False)
        for dof in acquisition_dofs:
            if dof.name not in points:
                raise ValueError(f"Cannot acquire points; missing values for {dof.name}.")

        n = len(points[dof.name])

        try:
            uid = yield from self.acquisition_plan(
                acquisition_dofs,
                points,
                [*self.detectors, *self.dofs.devices],
                delay=self.trigger_delay,
            )
            products = self.digestion(self.db[uid].table(fill=True), **self.digestion_kwargs)

        except KeyboardInterrupt as interrupt:
            raise interrupt

        except Exception as error:
            if not self.tolerate_acquisition_errors:
                raise error
            logging.warning(f"Error in acquisition/digestion: {repr(error)}")
            products = pd.DataFrame(points)
            for obj in self.objectives(active=True):
                products.loc[:, obj.name] = np.nan

        if len(products) != n:
            raise ValueError("The table returned by the digestion function must be the same length as the sampled inputs!")

        return products

    def load_data(self, data_file, append=True):
        new_table = pd.read_hdf(data_file, key="table")
        self._table = pd.concat([self._table, new_table]) if append else new_table
        self.refresh()

    def reset(self):
        """Reset the agent."""
        self._table = pd.DataFrame()

        for obj in self.objectives(active=True):
            if hasattr(obj, "_model"):
                del obj._model

        self.n_last_trained = 0

    def benchmark(
        self,
        output_dir="./",
        iterations=16,
        per_iter_learn_kwargs_list=[{"acqf": "qr", "n": 32}, {"acqf": "qei", "n": 1, "iterations": 4}],
    ):
        """Iterate over having the agent learn from scratch, and save the results to an output directory.

        Parameters
        ----------
        output_dir :
            Where to save the agent output.
        iterations : int
            How many benchmarks to run
        per_iter_learn_kwargs_list:
            A list of kwargs to pass to the agent.learn() method that the agent will run sequentially for each iteration.
        """

        for _ in range(iterations):
            self.reset()

            for kwargs in per_iter_learn_kwargs_list:
                yield from self.learn(**kwargs)

            self.save_data(f"{output_dir}/blop_benchmark_{int(ttime.time())}.h5")

    @property
    def model(self):
        """A model encompassing all the fitnesses and constraints."""
        active_objs = self.objectives(active=True)
        if all(hasattr(obj, "_model") for obj in active_objs):
            return ModelListGP(*[obj.model for obj in active_objs]) if len(active_objs) > 1 else active_objs[0].model
        raise ValueError("Not all active objectives have models.")

    def posterior(self, x):
        """A model encompassing all the objectives. A single GP in the single-objective case, or a model list."""
        return self.model.posterior(self.dofs(active=True).transform(torch.tensor(x)))

    @property
    def fitness_model(self):
        active_fitness_objectives = self.objectives(active=True, fitness=True)
        if len(active_fitness_objectives) == 0:
            # A dummy model that outputs noise, for when there are only constraints.
            dummy_X = self.sample(n=256, normalize=True).squeeze(-2)
            dummy_Y = torch.rand(size=(*dummy_X.shape[:-1], 1), dtype=torch.double)
            return construct_single_task_model(X=dummy_X, y=dummy_Y, min_noise=1e2, max_noise=2e2)
        if len(active_fitness_objectives) == 1:
            return active_fitness_objectives[0].model
        return ModelListGP(*[obj.model for obj in active_fitness_objectives])

    # @property
    # def pseudofitness_model(self):
    #     """
    #     In the case that we have all constraints, there is no fitness model. In that case,
    #     we replace the fitness model with a
    #     """
    #     active_fitness_objectives = self.objectives(active=True, fitness=True)
    #     if len(active_fitness_objectives) == 0:
    #         # A dummy model that outputs all ones, for when there are only constraints.
    #         dummy_X = self.sample(n=256, normalize=True).squeeze(-2)
    #         dummy_Y = torch.ones(size=(*dummy_X.shape[:-1], 1), dtype=torch.double)
    #         return construct_single_task_model(X=dummy_X, y=dummy_Y)
    #     if len(active_fitness_objectives) == 1:
    #         return active_fitness_objectives[0].model
    #     return ModelListGP(*[obj.model for obj in active_fitness_objectives])

    @property
    def evaluated_constraints(self):
        constraint_objectives = self.objectives(constraint=True)
        raw_targets_dict = self.raw_targets()
        if len(constraint_objectives):
            return torch.cat(
                [obj.constrain(raw_targets_dict[obj.name]).unsqueeze(-1) for obj in constraint_objectives], dim=-1
            )
        else:
            return torch.ones(size=(len(self._table), 0), dtype=torch.bool)

    def fitness_scalarization(self, weights="default"):
        active_fitness_objectives = self.objectives(active=True, fitness=True)
        if len(active_fitness_objectives) == 0:
            return ScalarizedPosteriorTransform(weights=torch.tensor([1.0], dtype=torch.double))
        if weights == "default":
            weights = torch.tensor([obj.weight for obj in active_fitness_objectives], dtype=torch.double)
        elif weights == "equal":
            weights = torch.ones(len(active_fitness_objectives), dtype=torch.double)
        elif weights == "random":
            weights = torch.rand(len(active_fitness_objectives), dtype=torch.double)
            weights *= len(active_fitness_objectives) / weights.sum()
        elif not isinstance(weights, torch.Tensor):
            raise ValueError(f"'weights' must be a Tensor or one of ['default', 'equal', 'random'], and not {weights}.")
        return ScalarizedPosteriorTransform(weights=weights)

    def scalarized_fitnesses(self, weights="default", constrained=True):
        """
        Return the scalar fitness for each sample, scalarized by the weighting scheme.

        If constrained=True, the points that satisfy the most constraints are automatically better than the others.
        """
        fitness_objs = self.objectives(fitness=True)
        if len(fitness_objs) >= 1:
            f = self.fitness_scalarization(weights=weights).evaluate(
                self.train_targets(active=True, fitness=True, concatenate=True)
            )
            f = torch.where(f.isnan(), -np.inf, f)  # remove all nans
        else:
            f = torch.zeros(len(self._table), dtype=torch.double)  # if there are no fitnesses, use a constant dummy fitness
        if constrained:
            # how many constraints are satisfied?
            c = self.evaluated_constraints.sum(axis=-1)
            f = torch.where(c < c.max(), -np.inf, f)
        return f

    def argmax_best_f(self, weights="default"):
        return int(self.scalarized_fitnesses(weights=weights, constrained=True).argmax())

    def best_f(self, weights="default"):
        return float(self.scalarized_fitnesses(weights=weights, constrained=True).max())

    @property
    def pareto_mask(self):
        """
        Returns a mask of all points that satisfy all constraints and are Pareto efficient.
        A point is Pareto efficient if it is there is no other point that is better at every objective.
        """
        Y = self.train_targets(active=True, fitness=True, concatenate=True)

        # nuke the bad points
        Y[~self.evaluated_constraints.all(axis=-1)] = -np.inf
        if Y.shape[-1] < 2:
            raise ValueError("Computing the Pareto front requires at least 2 fitness objectives.")
        in_pareto_front = ~(Y.unsqueeze(1) > Y.unsqueeze(0)).all(axis=-1).any(axis=0)
        return in_pareto_front & self.evaluated_constraints.all(axis=-1)

    @property
    def pareto_front(self):
        """
        A subset of the data table containing only points on the Pareto front.
        """
        return self._table.loc[self.pareto_mask.numpy()]

    @property
    def min_ref_point(self):
        y = self.train_targets(concatenate=True)[:, self.objectives.type == "fitness"]
        return y[y.argmax(axis=0)].min(axis=0).values

    @property
    def random_ref_point(self):
        return self.train_targets(active=True, fitness=True, concatenate=True)[self.argmax_best_f(weights="random")]

    @property
    def all_objectives_valid(self):
        """A mask of whether all objectives are valid for each data point."""
        return ~torch.isnan(self.scalarized_fitnesses())

    def _construct_model(self, obj, skew_dims=None):
        """
        Construct an untrained model for an objective.
        """

        skew_dims = skew_dims if skew_dims is not None else self._latent_dim_tuples(obj.name)

        train_inputs = self.train_inputs(active=True)
        train_targets = self.train_targets()[obj.name].unsqueeze(-1)

        inputs_are_trusted = ~torch.isnan(train_inputs).any(axis=1)
        targets_are_trusted = ~torch.isnan(train_targets).any(axis=1)

        trusted = inputs_are_trusted & targets_are_trusted & ~self.pruned_mask()

        obj._model = construct_single_task_model(
            X=train_inputs[trusted],
            y=train_targets[trusted],
            min_noise=obj.min_noise,
            max_noise=obj.max_noise,
            skew_dims=self._latent_dim_tuples()[obj.name],
        )

        obj.model_dofs = set(self.dofs(active=True).names)  # if these change, retrain the model on self.ask()

        if trusted.all():
            obj.validity_conjugate_model = None
            obj.validity_constraint = GenericDeterministicModel(f=lambda x: torch.ones(size=x.size())[..., -1])

        else:
            dirichlet_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                trusted.long(), learn_additional_noise=True
            )

            obj.validity_conjugate_model = models.LatentDirichletClassifier(
                train_inputs=train_inputs[inputs_are_trusted],
                train_targets=dirichlet_likelihood.transformed_targets.transpose(-1, -2)[inputs_are_trusted].double(),
                skew_dims=skew_dims,
                likelihood=dirichlet_likelihood,
                input_transform=self.input_normalization,
            )

            obj.validity_constraint = GenericDeterministicModel(
                f=lambda x: obj.validity_conjugate_model.probabilities(x)[..., -1]
            )

    def _construct_all_models(self):
        """Construct a model for each objective."""
        objectives_to_construct = self.objectives if self.model_inactive_objectives else self.objectives(active=True)
        for obj in objectives_to_construct:
            self._construct_model(obj)

    def _train_all_models(self, **kwargs):
        """Fit all of the agent's models. All kwargs are passed to `botorch.fit.fit_gpytorch_mll`."""
        t0 = ttime.monotonic()
        objectives_to_train = self.objectives if self.model_inactive_objectives else self.objectives(active=True)
        for obj in objectives_to_train:
            train_model(obj._model)
            if obj.validity_conjugate_model is not None:
                train_model(obj.validity_conjugate_model)

        if self.verbose:
            logger.info(f"trained models in {ttime.monotonic() - t0:.01f} seconds")

        self.n_last_trained = len(self._table)

    def _get_acquisition_function(self, identifier, return_metadata=False):
        """Returns a BoTorch acquisition function for a given identifier. Acquisition functions can be
        found in `agent.all_acqfs`.
        """

        return acquisition._construct_acqf(self, identifier=identifier, return_metadata=return_metadata)

    def _latent_dim_tuples(self, obj_index=None):
        """
        For the objective indexed by 'obj_index', return a list of tuples, where each tuple represents
        a group of DOFs to fit a latent representation to.
        """

        if obj_index is None:
            return {obj.name: self._latent_dim_tuples(obj_index=obj.name) for obj in self.objectives}

        obj = self.objectives[obj_index]

        latent_group_index = {}
        for dof in self.dofs(active=True):
            latent_group_index[dof.name] = dof.name
            for group_index, latent_group in enumerate(obj.latent_groups):
                if dof.name in latent_group:
                    latent_group_index[dof.name] = group_index

        u, uinv = np.unique(list(latent_group_index.values()), return_inverse=True)
        return [tuple(np.where(uinv == i)[0]) for i in range(len(u))]

    @property
    def sample_domain(self):
        """
        Returns a (2, n_active_dof) array of lower and upper bounds for dofs.
        Read-only DOFs are set to exactly their last known value.
        Discrete DOFs are relaxed to some continuous domain.
        """
        return self.dofs(active=True).transform(self.dofs(active=True).search_domain.T).clone()

    @property
    def input_normalization(self):
        """
        Suitably transforms model inputs to the unit hypercube.

        For modeling:

        Always normalize between min and max values. This is always inside the trust domain, sometimes smaller.

        For sampling:

        Settable: normalize between search bounds
        Read-only: constrain to the readback value
        """

        return Normalize(d=self.dofs.active.sum())

    def save_data(self, path="./data.h5"):
        """
        Save the sampled inputs and targets of the agent to a file, which can be used
        to initialize a future agent.
        """

        save_dir, _ = os.path.split(path)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._table.to_hdf(path, key="table")

    def forget(self, last=None, index=None, train=True):
        """
        Make the agent forget some data.

        Parameters
        ----------
        index :
            An index of samples to forget about.
        last : int
            Forget the last n=last points.
        """

        if last is not None:
            if last > len(self._table):
                raise ValueError(f"Cannot forget last {last} data points (only {len(self._table)} samples have been taken).")
            self.forget(index=self._table.index.values[-last:], train=train)

        elif index is not None:
            self._table.drop(index=index, inplace=True)
            self._construct_all_models()
            if train:
                self._train_all_models()

        else:
            raise ValueError("Must supply either 'last' or 'index'.")

    def _set_hypers(self, hypers):
        for obj in self.objectives(active=True):
            obj.model.load_state_dict(hypers[obj.name])
        self.validity_constraint.load_state_dict(hypers["validity_constraint"])

    def constraint(self, x):
        log_p = torch.zeros(x.shape[:-1])
        for obj in self.objectives(active=True):
            log_p += obj.log_total_constraint(x)

        return log_p.exp()  # + 1e-6 * normalize(x, self.sample_domain).square().sum(axis=-1)

    @property
    def hypers(self) -> dict:
        """Returns a dict of all the hyperparameters for each model in each objective."""
        hypers = {}
        for obj in self.objectives:
            hypers[obj.name] = {"model": {}, "validity_conjugate_model": {}}

            for key, value in obj.model.state_dict().items():
                hypers[obj.name]["model"][key] = value

            if obj.validity_conjugate_model is not None:
                for key, value in obj.validity_conjugate_model.state_dict().items():
                    hypers[obj.name]["validity_conjugate_model"][key] = value

        return hypers

    def save_hypers(self, filepath):
        """Save the agent's fitted hyperparameters to a given filepath."""
        hypers = self.hypers
        with h5py.File(filepath, "w") as f:
            for obj_name in hypers.keys():
                f.create_group(obj_name)
                f[obj_name].create_group("model")
                f[obj_name].create_group("validity_conjugate_model")

                for key, value in hypers[obj_name]["model"].items():
                    f[obj_name]["model"].create_dataset(key, data=value)

                for key, value in hypers[obj_name]["validity_conjugate_model"].items():
                    f[obj_name]["validity_conjugate_model"].create_dataset(key, data=value)

    @staticmethod
    def load_hypers(filepath) -> dict:
        """Load hyperparameters from a file."""
        hypers = {}
        with h5py.File(filepath, "r") as f:
            for obj_name in f.keys():
                hypers[obj_name] = {"model": OrderedDict(), "validity_conjugate_model": OrderedDict()}

                for key, value in f[obj_name]["model"].items():
                    hypers[obj_name]["model"][key] = torch.tensor(np.atleast_1d(value[()]))

                for key, value in f[obj_name]["validity_conjugate_model"].items():
                    hypers[obj_name]["validity_conjugate_model"][key] = torch.tensor(np.atleast_1d(value[()]))

        return hypers

    @property
    def all_acqfs(self):
        """
        Description and identifiers for all supported acquisition functions.
        """
        return acquisition.all_acqfs()

    def raw_inputs(self, index=None, **subset_kwargs):
        """
        Get the raw, untransformed inputs for a DOF (or for a subset).
        """
        if index is None:
            return torch.cat([self.raw_inputs(dof.name) for dof in self.dofs(**subset_kwargs)], dim=-1)
        return torch.tensor(self._table.loc[:, self.dofs[index].name].values, dtype=torch.double).unsqueeze(-1)

    def train_inputs(self, index=None, **subset_kwargs):
        """A two-dimensional tensor of all DOF values."""

        if index is None:
            return torch.cat([self.train_inputs(index=dof.name) for dof in self.dofs(**subset_kwargs)], dim=-1)

        dof = self.dofs[index]
        raw_inputs = self.raw_inputs(index=index, **subset_kwargs)

        # check that inputs values are inside acceptable values
        valid = (raw_inputs >= dof._trust_domain[0]) & (raw_inputs <= dof._trust_domain[1])
        raw_inputs = torch.where(valid, raw_inputs, np.nan)

        return dof._transform(raw_inputs)

    def raw_targets(self, index=None, **subset_kwargs):
        """
        Get the raw, untransformed inputs for an objective (or for a subset).
        """
        values = {}

        for obj in self.objectives(**subset_kwargs):
            # return torch.cat([self.raw_targets(index=obj.name) for obj in self.objectives(**subset_kwargs)], dim=-1)
            values[obj.name] = torch.tensor(self._table.loc[:, obj.name].values, dtype=torch.double)

        return values

    def train_targets(self, concatenate=False, **subset_kwargs):
        """Returns the values associated with an objective name."""

        targets_dict = {}
        raw_targets_dict = self.raw_targets(**subset_kwargs)

        for obj in self.objectives(**subset_kwargs):
            y = raw_targets_dict[obj.name]

            # check that targets values are inside acceptable values
            valid = (y >= obj._trust_domain[0]) & (y <= obj._trust_domain[1])
            y = torch.where(valid, y, np.nan)

            targets_dict[obj.name] = obj._transform(y)

        if self.enforce_all_objectives_valid:
            all_valid_mask = True

            for name, values in targets_dict.items():
                all_valid_mask &= ~values.isnan()

            for name in targets_dict.keys():
                targets_dict[name] = targets_dict[name].where(all_valid_mask, np.nan)

        if concatenate:
            return torch.cat([values.unsqueeze(-1) for values in targets_dict.values()], axis=-1)

        return targets_dict

    @property
    def best(self):
        """Returns all data for the best point."""
        return self._table.loc[self.argmax_best_f()]

    @property
    def best_inputs(self):
        """Returns the value of each DOF at the best point."""
        return self._table.loc[self.argmax_best_f(), self.dofs.names].to_dict()

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

        if len(self.dofs(active=True, read_only=False)) == 1:
            if len(self.objectives(active=True, fitness=True)) > 0:
                plotting._plot_fitness_objs_one_dof(self, **kwargs)
            if len(self.objectives(active=True, constraint=True)) > 0:
                plotting._plot_constraint_objs_one_dof(self, **kwargs)
        else:
            plotting._plot_objs_many_dofs(self, axes=axes, **kwargs)

    def plot_acquisition(self, acqf="ei", axes: Tuple = (0, 1), **kwargs):
        """Plot an acquisition function over test inputs sampling the limits of the parameter space.

        Parameters
        ----------
        acqf :
            Which acquisition function to plot. Can also take a list of acquisition functions.
        axes :
            A tuple specifying which DOFs to plot as a function of. Can be either an int or the name of DOFs.
        """
        if len(self.dofs(active=True, read_only=False)) == 1:
            plotting._plot_acqf_one_dof(self, acqfs=np.atleast_1d(acqf), **kwargs)

        else:
            plotting._plot_acqf_many_dofs(self, acqfs=np.atleast_1d(acqf), axes=axes, **kwargs)

    def plot_validity(self, axes: Tuple = (0, 1), **kwargs):
        """Plot the modeled constraint over test inputs sampling the limits of the parameter space.

        Parameters
        ----------
        axes :
            A tuple specifying which DOFs to plot as a function of. Can be either an int or the name of DOFs.
        """
        if len(self.dofs(active=True, read_only=False)) == 1:
            plotting._plot_valid_one_dof(self, **kwargs)

        else:
            plotting._plot_valid_many_dofs(self, axes=axes, **kwargs)

    def plot_history(self, **kwargs):
        """Plot the improvement of the agent over time."""
        plotting._plot_history(self, **kwargs)

    @property
    def latent_transforms(self):
        return {obj.name: obj.model.covar_module.latent_transform for obj in self.objectives(active=True)}

    def plot_pareto_front(self, **kwargs):
        """Plot the improvement of the agent over time."""
        plotting._plot_pareto_front(self, **kwargs)

    def prune(self, pruning_objs=[], thresholds=[]):
        """Prune low-fidelity datapoints from model fitting"""
        # set the prune column to false
        self._table = self._table.assign(prune=[False for i in range(self._table.shape[0])])
        # make sure there are models trained for all the objectives we are pruning over
        if not all(hasattr(obj, "model") for obj in pruning_objs):
            raise ValueError("Not all pruning objectives have models.")
        # make sure we have the same number of thresholds and objectives to prune over
        if len(pruning_objs) != len(thresholds):
            raise ValueError("Number of pruning objectives and thresholds should be the same")
        for i in range(len(pruning_objs)):
            obj = pruning_objs[i]
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(obj.model.likelihood, obj.model)
            mlls = mll(obj.model(self.train_inputs()), self.train_targets()[obj.name].unsqueeze(-1)).detach()
            mlls -= mlls.max()
            mlls_wo_nans = [x for x in mlls if not np.isnan(x)]
            # Q: SHOULD WE MAKE AN OPTION TO HAVE THIS BE >, IN CASE THEY ARE NOT NEGATED?
            if len(mlls_wo_nans) > 0:
                self._table["prune"] = torch.logical_or(
                    torch.tensor(self._table["prune"].values), mlls < thresholds[i] * np.quantile(mlls_wo_nans, q=0.25)
                )
        self.refresh()
        # return self._table["prune"]

    # @property
    def pruned_mask(self):
        if self.exclude_pruned and "prune" in self._table.columns:
            return torch.tensor(self._table.prune.values.astype(bool))
        return torch.zeros(len(self._table)).bool()
