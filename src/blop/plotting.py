import warnings

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from .bayesian.acquisition import _construct_acqf, parse_acqf_identifier

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
# Note: the values near 1 are hard to see on a white background. Turbo goes from red to blue and isn't white in the middle.
DEFAULT_COLORMAP = "turbo"
DEFAULT_SCATTER_SIZE = 16

MAX_TEST_INPUTS = 2**11

warnings.warn("The plotting module is deprecated and will be removed in Blop v1.0.0.", DeprecationWarning, stacklevel=2)


def _plot_objs_one_dof(agent, size=16, lw=1e0):
    fitness_objs = agent.objectives(fitness=True)

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(fitness_objs),
        3,
        figsize=(12, 4 * len(agent.objectives)),
        sharex=True,
        constrained_layout=True,
    )

    agent.obj_axes = np.atleast_2d(agent.obj_axes)

    x_dof = agent.dofs(active=True)[0]
    x_values = agent.table[x_dof.movable.name]

    test_inputs = agent.sample(n=256, method="grid")
    test_model_inputs = agent.dofs.transform(test_inputs)
    test_x = test_inputs[..., 0].detach().numpy()

    for obj_index, obj in enumerate(agent.objectives):
        color = DEFAULT_COLOR_LIST[obj_index]

        obj_values = agent.train_targets(index=obj.name).numpy()

        if obj.target is not None:
            test_posterior = obj.model.posterior(test_model_inputs)
            test_mean = test_posterior.mean[..., 0, obj_index].detach().numpy()

            test_sigma = test_posterior.variance.sqrt()[..., 0, obj_index].detach().numpy()
            agent.obj_axes[obj_index, 0].scatter(x_values, obj_values, s=size, color=color)

            for z in [0, 1, 2]:
                agent.obj_axes[obj_index, 0].fill_between(
                    test_x.ravel(),
                    (test_mean - z * test_sigma).ravel(),
                    (test_mean + z * test_sigma).ravel(),
                    lw=lw,
                    color=color,
                    alpha=0.5**z,
                )

            agent.obj_axes[obj_index, 0].set_xlim(*x_dof.search_domain)
            agent.obj_axes[obj_index, 0].set_xlabel(x_dof.label_with_units)
            agent.obj_axes[obj_index, 0].set_ylabel(obj.label_with_units)

        if obj.constraint is not None:
            test_constraint_prob = obj.constraint_probability(test_model_inputs).detach().squeeze()
        else:
            test_constraint_prob = torch.ones(test_x.shape)

        agent.obj_axes[obj_index, 1].plot(test_x.ravel(), test_constraint_prob.ravel(), lw=lw, color=color)

        if obj.validity_conjugate_model and obj.validity_probability:
            test_valid_prob = obj.validity_probability(test_model_inputs).detach().squeeze()
        else:
            test_valid_prob = torch.ones(test_x.shape)

        agent.obj_axes[obj_index, 2].plot(test_x.ravel(), test_valid_prob.ravel(), lw=lw, color=color)

        agent.obj_axes[obj_index, 0].set_title("posterior")
        agent.obj_axes[obj_index, 1].set_title("constraint")
        agent.obj_axes[obj_index, 2].set_title("validity")

        for ax in agent.obj_axes[obj_index]:
            ax.set_xlabel(x_dof.label_with_units)
            ax.set_xlim(*x_dof.search_domain)

        agent.obj_axes[obj_index, 0].set_ylabel(obj.label_with_units)
        agent.obj_axes[obj_index, 1].set_ylabel("constraint probability")
        agent.obj_axes[obj_index, 2].set_ylabel("validity probability")


def _plot_objs_many_dofs(
    agent,
    axes: tuple[int] = (0, 1),
    gridded: bool = False,
    n: int = 1024,
    cmap: str = DEFAULT_COLORMAP,
    size: float = 16,
):
    """
    Axes represents which active, non-read-only axes to plot with
    """

    plottable_dofs = agent.dofs(active=True, read_only=False)

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(agent.objectives),
        5,
        figsize=(12, 3 * len(agent.objectives)),
        constrained_layout=True,
        dpi=256,
    )

    agent.obj_axes = np.atleast_2d(agent.obj_axes)

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    x_values = agent.table[x_dof.movable.name]
    y_values = agent.table[y_dof.movable.name]

    # test_inputs has shape (*input_shape, 1, n_active_dofs)
    # test_x and test_y should be squeezeable
    test_inputs = agent.sample(n=n, method="grid") if gridded else agent.sample(n=n)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    test_model_inputs = agent.dofs.transform(test_inputs)

    for obj_index, obj in enumerate(agent.objectives):
        targets = agent.train_targets(index=obj.name)[:, 0]

        values = obj._untransform(targets).numpy()
        # mask does not generate properly when values is a tensor (returns values of 0 instead of booleans)

        val_vmin, val_vmax = np.nanquantile(values, q=[0.01, 0.99])
        val_norm = (
            mpl.colors.LogNorm(val_vmin, val_vmax) if obj.transform == "log" else mpl.colors.Normalize(val_vmin, val_vmax)
        )

        # mask for nan values, uses unfilled o marker
        mask = np.isnan(values)
        values_ax = agent.obj_axes[obj_index, 0].scatter(
            np.array(x_values)[~mask], np.array(y_values)[~mask], c=values[~mask], s=size, norm=val_norm, cmap=cmap
        )
        agent.obj_axes[obj_index, 0].scatter(
            np.array(x_values)[mask], np.array(y_values)[mask], marker="o", ec="k", fc="w", s=size
        )

        # mean and sigma will have shape (*input_shape,)
        test_posterior = obj.model.posterior(test_model_inputs)
        test_mean = test_posterior.mean[..., 0, 0].detach().squeeze()
        test_sigma = test_posterior.variance.sqrt()[..., 0, 0].detach().squeeze()

        if obj.constraint is not None:
            test_constraint_prob = obj.constraint_probability(test_model_inputs)[..., 0].squeeze()
        else:
            test_constraint_prob = torch.ones((len(test_x), len(test_y))) if gridded else torch.ones(len(test_x))

        if not obj.all_valid:
            test_valid_prob = obj.validity_probability(test_model_inputs)[..., 0].squeeze()
        else:
            test_valid_prob = torch.ones((len(test_x), len(test_y))) if gridded else torch.ones(len(test_x))

        if gridded:
            post_mean_ax = agent.obj_axes[obj_index, 1].pcolormesh(
                test_x,
                test_y,
                obj._untransform(test_mean),
                shading="nearest",
                norm=val_norm,
                cmap=cmap,
            )
            post_sigma_ax = agent.obj_axes[obj_index, 2].pcolormesh(
                test_x,
                test_y,
                test_sigma,
                shading="nearest",
                cmap=cmap,
                norm=mpl.colors.LogNorm(),
            )

            constraint_prob_ax = agent.obj_axes[obj_index, 3].pcolormesh(
                test_x,
                test_y,
                test_constraint_prob,
                shading="nearest",
                cmap=cmap,
            )

            valid_prob_ax = agent.obj_axes[obj_index, 4].pcolormesh(
                test_x,
                test_y,
                test_valid_prob,
                shading="nearest",
                cmap=cmap,
            )

        else:
            post_mean_ax = agent.obj_axes[obj_index, 1].scatter(
                test_x,
                test_y,
                c=obj._untransform(test_mean),
                s=size,
                norm=val_norm,
                cmap=cmap,
            )
            post_sigma_ax = agent.obj_axes[obj_index, 2].scatter(
                test_x,
                test_y,
                c=test_sigma,
                s=size,
                cmap=cmap,
                norm=mpl.colors.LogNorm(),
            )

            constraint_prob_ax = agent.obj_axes[obj_index, 3].scatter(
                test_x,
                test_y,
                c=test_constraint_prob,
                s=size,
                cmap=cmap,
            )

            valid_prob_ax = agent.obj_axes[obj_index, 4].scatter(
                test_x,
                test_y,
                c=test_valid_prob,
                s=size,
                cmap=cmap,
            )

        cbars = {}
        cbars["values"] = agent.obj_fig.colorbar(
            values_ax, ax=agent.obj_axes[obj_index, 0], location="bottom", aspect=32, shrink=0.8
        )
        # cbars["values"].set_label(f"{obj.units or ''}")
        cbars["post_mean"] = agent.obj_fig.colorbar(
            post_mean_ax, ax=agent.obj_axes[obj_index, 1], location="bottom", aspect=32, shrink=0.8
        )
        cbars["post_sigma"] = agent.obj_fig.colorbar(
            post_sigma_ax, ax=agent.obj_axes[obj_index, 2], location="bottom", aspect=32, shrink=0.8
        )
        cbars["constraint"] = agent.obj_fig.colorbar(
            constraint_prob_ax, ax=agent.obj_axes[obj_index, 3], location="bottom", aspect=32, shrink=0.95
        )
        cbars["validity"] = agent.obj_fig.colorbar(
            valid_prob_ax, ax=agent.obj_axes[obj_index, 4], location="bottom", aspect=32, shrink=0.95
        )
        # constraint_cbar.set_label(f"{obj.name} constraint")

        for cbar_name, cbar in cbars.items():
            cbar.ax.minorticks_off()
            # cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: locale.format_string('%.1f', x)))
            # cbar.set_ticklabels(cbar.ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')

            vmin = cbars[cbar_name].norm.vmin
            vmax = cbars[cbar_name].norm.vmax

            if isinstance(cbar.norm, mpl.colors.LogNorm):
                ticks = np.geomspace(vmin, vmax, 3)
            else:
                ticks = np.linspace(vmin, vmax, 3)

            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{x:.01e}" for x in cbar.get_ticks()])

        col_names = [
            "samples",
            "post. mean",
            "post. raw std. dev.",
            "constraint",
            "validity",
        ]

        pad = 5

        for column_index, ax in enumerate(agent.obj_axes[obj_index]):
            ax.set_title(
                col_names[column_index],
            )

    for row_index, ax in enumerate(agent.obj_axes[:, 0]):
        ax.annotate(
            agent.objectives[row_index].name,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            color="k",
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
        )

    for ax in agent.obj_axes[:, 0]:
        ax.set_ylabel(y_dof.label_with_units)

    for ax in agent.obj_axes.ravel():
        ax.set_xlabel(x_dof.label_with_units)

        ax.set_xlim(*x_dof.search_domain)
        ax.set_ylim(*y_dof.search_domain)
        if x_dof.transform == "log":
            ax.set_xscale("log")
        if y_dof.transform == "log":
            ax.set_yscale("log")


def _plot_acqf_one_dof(agent, acqfs, lw=1e0, **kwargs):
    agent.acq_fig, agent.acq_axes = plt.subplots(
        1,
        len(acqfs),
        figsize=(4 * len(acqfs), 4),
        sharex=True,
        constrained_layout=True,
    )

    agent.acq_axes = np.atleast_1d(agent.acq_axes)
    x_dof = agent.dofs(active=True)[0]

    test_inputs = agent.sample(method="grid")
    model_inputs = agent.dofs.transform(test_inputs)

    for iacqf, acqf_identifier in enumerate(acqfs):
        color = DEFAULT_COLOR_LIST[iacqf]

        acqf_config = parse_acqf_identifier(acqf_identifier)
        acqf, _ = _construct_acqf(agent, acqf_config["name"])

        test_acqf_value = acqf(model_inputs).detach().numpy()

        agent.acq_axes[iacqf].plot(test_inputs.squeeze(-2), test_acqf_value, lw=lw, color=color)

        agent.acq_axes[iacqf].set_xlim(*x_dof.search_domain)
        agent.acq_axes[iacqf].set_xlabel(x_dof.label_with_units)
        agent.acq_axes[iacqf].set_ylabel(acqf_config["name"])


def _plot_acqf_many_dofs(
    agent, acqfs, axes=(0, 1), shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16, **kwargs
):
    agent.acq_fig, agent.acq_axes = plt.subplots(
        1,
        len(acqfs),
        figsize=(4 * len(acqfs), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    plottable_dofs = agent.dofs(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    agent.acq_axes = np.atleast_1d(agent.acq_axes)

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    # test_inputs has shape (..., 1, n_active_dofs)
    test_inputs = agent.sample(n=1024, method="grid") if gridded else agent.sample(n=1024)
    model_inputs = agent.dofs.transform(test_inputs)
    *test_dim, input_dim = test_inputs.shape
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    for iacqf, acqf_identifier in enumerate(acqfs):
        acqf_config = parse_acqf_identifier(acqf_identifier)
        acqf, _ = _construct_acqf(agent, acqf_config["name"])

        test_acqf_value = acqf(model_inputs.reshape(-1, 1, input_dim)).detach().reshape(test_dim).squeeze().numpy()

        if gridded:
            agent.acq_axes[iacqf].set_title(acqf_config["name"])
            obj_ax = agent.acq_axes[iacqf].pcolormesh(
                test_x,
                test_y,
                test_acqf_value,
                shading=shading,
                cmap=cmap,
            )

            agent.acq_fig.colorbar(obj_ax, ax=agent.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

        else:
            agent.acq_axes[iacqf].set_title(acqf_config["name"])
            obj_ax = agent.acq_axes[iacqf].scatter(
                test_x,
                test_y,
                c=test_acqf_value,
            )

            agent.acq_fig.colorbar(obj_ax, ax=agent.acq_axes[iacqf], location="bottom", aspect=32, shrink=0.8)

    for ax in agent.acq_axes.ravel():
        ax.set_xlabel(x_dof.label_with_units)
        ax.set_ylabel(y_dof.label_with_units)
        ax.set_xlim(*x_dof.search_domain)
        ax.set_ylim(*y_dof.search_domain)
        if x_dof.transform == "log":
            ax.set_xscale("log")
        if y_dof.transform == "log":
            ax.set_yscale("log")


def _plot_valid_one_dof(agent, size=16, lw=1e0):
    agent.valid_fig, agent.valid_ax = plt.subplots(1, 1, figsize=(6, 4 * len(agent.objectives)), constrained_layout=True)

    x_dof = agent.dofs(active=True)[0]
    x_values = agent.table[x_dof.movable.name]

    test_inputs = agent.sample(method="grid")
    constraint = agent.constraint(agent.dofs.transform(test_inputs))[..., 0]

    agent.valid_ax.scatter(x_values, agent.all_objectives_valid, s=size)
    agent.valid_ax.plot(test_inputs.squeeze(-2), constraint, lw=lw)
    agent.valid_ax.set_xlim(*x_dof.search_domain)


def _plot_valid_many_dofs(agent, axes=(0, 1), shading="nearest", cmap=DEFAULT_COLORMAP, size=16, gridded=None):
    agent.valid_fig, agent.valid_axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    plottable_dofs = agent.dofs(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    # test_inputs has shape (..., 1, n_active_dofs)
    test_inputs = agent.sample(method="grid") if gridded else agent.sample(n=1024)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    constraint = agent.constraint(agent.dofs.transform(test_inputs))[..., 0].squeeze().numpy()

    if gridded:
        _ = agent.valid_axes[1].pcolormesh(
            test_x,
            test_y,
            constraint,
            shading=shading,
            cmap=cmap,
            vmin=0,
            vmax=0,
        )

    else:
        _ = agent.valid_axes[1].scatter(
            test_x,
            test_y,
            c=constraint,
            s=size,
            cmap=cmap,
            vmin=0,
            vmax=0,
        )

    for ax in agent.valid_axes.ravel():
        ax.set_xlabel(x_dof.label_with_units)
        ax.set_ylabel(y_dof.label_with_units)
        ax.set_xlim(*x_dof.search_domain)
        ax.set_ylim(*y_dof.search_domain)
        if x_dof.transform == "log":
            ax.set_xscale("log")
        if y_dof.transform == "log":
            ax.set_yscale("log")


def _plot_history(agent, x_key="index", show_all_objs=False):
    x = list(range(0, len(agent.table[next(iter(agent.table))])))

    num_obj_plots = 1
    if show_all_objs:
        num_obj_plots = len(agent.objectives) + 1

    len(agent.objectives) + 1 if len(agent.objectives) > 1 else 1

    _, hist_axes = plt.subplots(
        num_obj_plots, 1, figsize=(6, 4 * num_obj_plots), sharex=True, constrained_layout=True, dpi=200
    )
    hist_axes = np.atleast_1d(hist_axes)

    unique_strategies, _, acqf_inverse = np.unique(agent.table["acqf"], return_index=True, return_inverse=True)

    sample_colors = np.array(DEFAULT_COLOR_LIST)[acqf_inverse]

    if show_all_objs:
        for obj_index, obj in enumerate(agent.objectives):
            y = agent.table.loc[:, f"{obj.key}_fitness"].values
            hist_axes[obj_index].scatter(x, y, c=sample_colors)
            hist_axes[obj_index].plot(x, y, lw=5e-1, c="k")
            hist_axes[obj_index].set_ylabel(obj.key)

    y = agent.scalarized_fitnesses()

    cummax_y = np.array([np.nanmax(y[: i + 1]) for i in range(len(y))])

    hist_axes[-1].scatter(x, y, c=sample_colors)
    hist_axes[-1].plot(x, y, lw=5e-1, c="k")

    hist_axes[-1].plot(x, cummax_y, lw=5e-1, c="k", ls=":")

    hist_axes[-1].set_ylabel("total_fitness")
    hist_axes[-1].set_xlabel(x_key)

    handles = []
    for i_acqf, acqf in enumerate(unique_strategies):
        handles.append(Patch(color=DEFAULT_COLOR_LIST[i_acqf], label=acqf))
    legend = hist_axes[0].legend(handles=handles, fontsize=8)
    legend.set_title("acquisition function")


def inspect_beam(agent, index, border=None):
    im = agent.images[index]

    x_min, x_max, y_min, y_max, width_x, width_y = agent.table.loc[
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


def _plot_pareto_front(agent, obj_indices=(0, 1)):
    f_objs = agent.objectives(fitness=True)
    (i, j) = obj_indices

    if len(f_objs) < 2:
        raise ValueError("Cannot plot Pareto front for agents with fewer than two fitness objectives")

    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    y = agent.train_targets(fitness=True)

    pareto_mask = agent.pareto_mask
    constraint = agent.evaluated_constraints.all(axis=-1)

    ax.scatter(*y[(~pareto_mask) & constraint].T[[i, j]], c="k")
    ax.scatter(*y[~constraint].T[[i, j]], c="r", marker="x", label="invalid")
    ax.scatter(*y[pareto_mask].T[[i, j]], c="b", label="Pareto front")

    ax.set_xlabel(f"{f_objs[i].name} fitness")
    ax.set_ylabel(f"{f_objs[j].name} fitness")

    ax.legend()
