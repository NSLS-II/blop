import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from . import acquisition

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
DEFAULT_COLORMAP = "magma"
DEFAULT_SCATTER_SIZE = 16

MAX_TEST_INPUTS = 2**11


def _plot_objs_one_dof(agent, size=16, lw=1e0):
    agent.obj_fig, agent.obj_axes = plt.subplots(
        agent.n_objs,
        1,
        figsize=(6, 4 * agent.n_objs),
        sharex=True,
        constrained_layout=True,
    )

    agent.obj_axes = np.atleast_1d(agent.obj_axes)

    x_dof = agent.dofs.subset(active=True)[0]
    x_values = agent.table.loc[:, x_dof.device.name].values

    for obj_index, obj in enumerate(agent.objectives):
        obj_fitness = agent._get_objective_targets(obj_index)

        color = DEFAULT_COLOR_LIST[obj_index]

        test_inputs = agent.test_inputs_grid()
        test_x = test_inputs[..., 0].detach().numpy()

        test_posterior = obj.model.posterior(test_inputs)
        test_mean = test_posterior.mean[..., 0].detach().numpy()
        test_sigma = test_posterior.variance.sqrt()[..., 0].detach().numpy()

        agent.obj_axes[obj_index].scatter(x_values, obj_fitness, s=size, color=color)

        for z in [0, 1, 2]:
            agent.obj_axes[obj_index].fill_between(
                test_x.ravel(),
                (test_mean - z * test_sigma).ravel(),
                (test_mean + z * test_sigma).ravel(),
                lw=lw,
                color=color,
                alpha=0.5**z,
            )

        agent.obj_axes[obj_index].set_xlim(*x_dof.limits)
        agent.obj_axes[obj_index].set_xlabel(x_dof.label)
        agent.obj_axes[obj_index].set_ylabel(obj.label)


def _plot_objs_many_dofs(agent, axes=(0, 1), shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=32, grid_zoom=1):
    """
    Axes represents which active, non-read-only axes to plot with
    """

    plottable_dofs = agent.dofs.subset(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(agent.objectives),
        3,
        figsize=(10, 4 * len(agent.objectives)),
        constrained_layout=True,
        dpi=256,
    )

    agent.obj_axes = np.atleast_2d(agent.obj_axes)

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    x_values = agent.table.loc[:, x_dof.device.name].values
    y_values = agent.table.loc[:, y_dof.device.name].values

    # test_inputs has shape (*input_shape, 1, n_active_dofs)
    # test_x and test_y should be squeezeable
    test_inputs = agent.test_inputs_grid() if gridded else agent.test_inputs(n=1024)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    for obj_index, obj in enumerate(agent.objectives):
        obj_values = agent._get_objective_targets(obj_index)

        obj_vmin, obj_vmax = np.nanpercentile(obj_values, q=[1, 99])
        obj_norm = mpl.colors.Normalize(obj_vmin, obj_vmax)

        data_ax = agent.obj_axes[obj_index, 0].scatter(x_values, y_values, c=obj_values, s=size, norm=obj_norm, cmap=cmap)

        # mean and sigma will have shape (*input_shape,)
        test_posterior = obj.model.posterior(test_inputs)
        test_mean = test_posterior.mean[..., 0, 0].detach().squeeze().numpy()
        test_sigma = test_posterior.variance.sqrt()[..., 0, 0].detach().squeeze().numpy()

        if gridded:
            _ = agent.obj_axes[obj_index, 1].pcolormesh(
                test_x,
                test_y,
                test_mean,
                shading=shading,
                cmap=cmap,
                norm=obj_norm,
            )
            sigma_ax = agent.obj_axes[obj_index, 2].pcolormesh(
                test_x,
                test_y,
                test_sigma,
                shading=shading,
                cmap=cmap,
                norm=mpl.colors.LogNorm(),
            )

        else:
            _ = agent.obj_axes[obj_index, 1].scatter(
                test_x,
                test_y,
                c=test_mean,
                s=size,
                norm=obj_norm,
                cmap=cmap,
            )
            sigma_ax = agent.obj_axes[obj_index, 2].scatter(
                test_x,
                test_y,
                c=test_sigma,
                s=size,
                cmap=cmap,
                norm=mpl.colors.LogNorm(),
            )

        obj_cbar = agent.obj_fig.colorbar(
            data_ax, ax=agent.obj_axes[obj_index, :2], location="bottom", aspect=32, shrink=0.8
        )
        err_cbar = agent.obj_fig.colorbar(
            sigma_ax, ax=agent.obj_axes[obj_index, 2], location="bottom", aspect=32, shrink=0.8
        )
        obj_cbar.set_label(obj.label)
        err_cbar.set_label(f"{obj.label} error")

    col_names = ["samples", "posterior mean", "posterior std. dev."]

    pad = 5

    for column_index, ax in enumerate(agent.obj_axes[0]):
        ax.annotate(
            col_names[column_index],
            xy=(0.5, 1),
            xytext=(0, pad),
            color="k",
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    if agent.n_objs > 1:
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

    for ax in agent.obj_axes.ravel():
        ax.set_xlabel(x_dof.label)
        ax.set_ylabel(y_dof.label)
        ax.set_xlim(*x_dof.limits)
        ax.set_ylim(*y_dof.limits)


def _plot_acqf_one_dof(agent, acq_funcs, lw=1e0, **kwargs):
    agent.acq_fig, agent.acq_axes = plt.subplots(
        1,
        len(acq_funcs),
        figsize=(4 * len(acq_funcs), 4),
        sharex=True,
        constrained_layout=True,
    )

    agent.acq_axes = np.atleast_1d(agent.acq_axes)
    x_dof = agent.dofs.subset(active=True)[0]

    test_inputs = agent.test_inputs_grid()

    for iacq_func, acq_func_identifier in enumerate(acq_funcs):
        color = DEFAULT_COLOR_LIST[iacq_func]

        acq_func, acq_func_meta = acquisition.get_acquisition_function(agent, acq_func_identifier)
        test_acqf = acq_func(test_inputs).detach().numpy()

        agent.acq_axes[iacq_func].plot(test_inputs.squeeze(-2), test_acqf, lw=lw, color=color)

        agent.acq_axes[iacq_func].set_xlim(*x_dof.limits)
        agent.acq_axes[iacq_func].set_xlabel(x_dof.label)
        agent.acq_axes[iacq_func].set_ylabel(acq_func_meta["name"])


def _plot_acqf_many_dofs(
    agent, acq_funcs, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16, **kwargs
):
    agent.acq_fig, agent.acq_axes = plt.subplots(
        1,
        len(acq_funcs),
        figsize=(4 * len(acq_funcs), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    plottable_dofs = agent.dofs.subset(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    agent.acq_axes = np.atleast_1d(agent.acq_axes)

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    # test_inputs has shape (..., 1, n_active_dofs)
    test_inputs = agent.test_inputs_grid() if gridded else agent.test_inputs(n=1024)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    for iacq_func, acq_func_identifier in enumerate(acq_funcs):
        acq_func, acq_func_meta = acquisition.get_acquisition_function(agent, acq_func_identifier)

        test_acqf = acq_func(test_inputs).detach().squeeze().numpy()

        if gridded:
            agent.acq_axes[iacq_func].set_title(acq_func_meta["name"])
            obj_ax = agent.acq_axes[iacq_func].pcolormesh(
                test_x,
                test_y,
                test_acqf,
                shading=shading,
                cmap=cmap,
            )

            agent.acq_fig.colorbar(obj_ax, ax=agent.acq_axes[iacq_func], location="bottom", aspect=32, shrink=0.8)

        else:
            agent.acq_axes[iacq_func].set_title(acq_func_meta["name"])
            obj_ax = agent.acq_axes[iacq_func].scatter(
                test_x,
                test_y,
                c=test_acqf,
            )

            agent.acq_fig.colorbar(obj_ax, ax=agent.acq_axes[iacq_func], location="bottom", aspect=32, shrink=0.8)

    for ax in agent.acq_axes.ravel():
        ax.set_xlabel(x_dof.label)
        ax.set_ylabel(y_dof.label)
        ax.set_xlim(*x_dof.limits)
        ax.set_ylim(*y_dof.limits)


def _plot_valid_one_dof(agent, size=16, lw=1e0):
    agent.valid_fig, agent.valid_ax = plt.subplots(1, 1, figsize=(6, 4 * agent.n_objs), constrained_layout=True)

    x_dof = agent.dofs.subset(active=True)[0]
    x_values = agent.table.loc[:, x_dof.device.name].values

    test_inputs = agent.test_inputs_grid()
    constraint = agent.constraint(test_inputs)[..., 0]

    agent.valid_ax.scatter(x_values, agent.all_objectives_valid, s=size)
    agent.valid_ax.plot(test_inputs.squeeze(-2), constraint, lw=lw)
    agent.valid_ax.set_xlim(*x_dof.limits)


def _plot_valid_many_dofs(agent, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=16, gridded=None):
    agent.valid_fig, agent.valid_axes = plt.subplots(1, 2, figsize=(8, 4 * agent.n_objs), constrained_layout=True)

    plottable_dofs = agent.dofs.subset(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    # test_inputs has shape (..., 1, n_active_dofs)
    test_inputs = agent.test_inputs_grid() if gridded else agent.test_inputs(n=1024)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    constraint = agent.constraint(test_inputs)[..., 0].squeeze().numpy()

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

    for ax in agent.acq_axes.ravel():
        ax.set_xlabel(x_dof.label)
        ax.set_ylabel(y_dof.label)
        ax.set_xlim(*x_dof.limits)
        ax.set_ylim(*y_dof.limits)


def _plot_history(agent, x_key="index", show_all_objs=False):
    x = getattr(agent.table, x_key).values

    num_obj_plots = 1
    if show_all_objs:
        num_obj_plots = agent.n_objs + 1

    agent.n_objs + 1 if agent.n_objs > 1 else 1

    hist_fig, hist_axes = plt.subplots(
        num_obj_plots, 1, figsize=(6, 4 * num_obj_plots), sharex=True, constrained_layout=True, dpi=200
    )
    hist_axes = np.atleast_1d(hist_axes)

    unique_strategies, acq_func_index, acq_func_inverse = np.unique(
        agent.table.acq_func, return_index=True, return_inverse=True
    )

    sample_colors = np.array(DEFAULT_COLOR_LIST)[acq_func_inverse]

    if show_all_objs:
        for obj_index, obj in enumerate(agent.objectives):
            y = agent.table.loc[:, f"{obj.key}_fitness"].values
            hist_axes[obj_index].scatter(x, y, c=sample_colors)
            hist_axes[obj_index].plot(x, y, lw=5e-1, c="k")
            hist_axes[obj_index].set_ylabel(obj.key)

    y = agent.scalarized_objectives

    cummax_y = np.array([np.nanmax(y[: i + 1]) for i in range(len(y))])

    hist_axes[-1].scatter(x, y, c=sample_colors)
    hist_axes[-1].plot(x, y, lw=5e-1, c="k")

    hist_axes[-1].plot(x, cummax_y, lw=5e-1, c="k", ls=":")

    hist_axes[-1].set_ylabel("total_fitness")
    hist_axes[-1].set_xlabel(x_key)

    handles = []
    for i_acq_func, acq_func in enumerate(unique_strategies):
        handles.append(Patch(color=DEFAULT_COLOR_LIST[i_acq_func], label=acq_func))
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
