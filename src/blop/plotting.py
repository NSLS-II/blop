import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from .bayesian.acquisition import _construct_acqf, parse_acqf_identifier

DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
# Note: the values near 1 are hard to see on a white background. Turbo goes from red to blue and isn't white in the middle.
DEFAULT_COLORMAP = "turbo"
DEFAULT_SCATTER_SIZE = 16

MAX_TEST_INPUTS = 2**11


def _plot_fitness_objs_one_dof(agent, size=16, lw=1e0):
    fitness_objs = agent.objectives(fitness=True)

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(fitness_objs),
        1,
        figsize=(6, 4 * len(fitness_objs)),
        sharex=True,
        constrained_layout=True,
    )

    agent.obj_axes = np.atleast_1d(agent.obj_axes)

    x_dof = agent.dofs(active=True)[0]
    x_values = agent.table.loc[:, x_dof.device.name].values

    test_inputs = agent.sample(method="grid")
    test_model_inputs = agent.dofs(active=True).transform(test_inputs)

    for obj_index, obj in enumerate(fitness_objs):
        obj_values = agent.train_targets()[obj.name].numpy()

        color = DEFAULT_COLOR_LIST[obj_index]

        test_inputs = agent.sample(method="grid")
        test_x = test_inputs[..., 0].detach().numpy()

        test_posterior = obj.model.posterior(test_model_inputs)
        test_mean = test_posterior.mean[..., 0].detach().numpy()
        test_sigma = test_posterior.variance.sqrt()[..., 0].detach().numpy()

        agent.obj_axes[obj_index].scatter(x_values, obj_values, s=size, color=color)

        for z in [0, 1, 2]:
            agent.obj_axes[obj_index].fill_between(
                test_x.ravel(),
                (test_mean - z * test_sigma).ravel(),
                (test_mean + z * test_sigma).ravel(),
                lw=lw,
                color=color,
                alpha=0.5**z,
            )

        agent.obj_axes[obj_index].set_xlim(*x_dof.search_domain)
        agent.obj_axes[obj_index].set_xlabel(x_dof.label_with_units)
        agent.obj_axes[obj_index].set_ylabel(obj.label_with_units)


def _plot_constraint_objs_one_dof(agent, size=16, lw=1e0):
    constraint_objs = agent.objectives(kind="constraint")

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(constraint_objs),
        2,
        figsize=(8, 4 * len(constraint_objs)),
        sharex=True,
        constrained_layout=True,
    )

    agent.obj_axes = np.atleast_2d(agent.obj_axes)

    x_dof = agent.dofs(active=True)[0]
    x_values = agent.table.loc[:, x_dof.device.name].values

    test_inputs = agent.sample(method="grid")
    test_model_inputs = agent.dofs(active=True).transform(test_inputs)

    for obj_index, obj in enumerate(constraint_objs):
        val_ax = agent.obj_axes[obj_index, 0]
        con_ax = agent.obj_axes[obj_index, 1]

        obj_values = agent.train_targets()[obj.name].numpy()

        color = DEFAULT_COLOR_LIST[obj_index]

        test_inputs = agent.sample(method="grid")
        test_x = test_inputs[..., 0].detach().numpy()

        test_posterior = obj.model.posterior(test_model_inputs)
        test_mean = test_posterior.mean[..., 0].detach().numpy()
        test_sigma = test_posterior.variance.sqrt()[..., 0].detach().numpy()

        val_ax.scatter(x_values, obj_values, s=size, color=color)

        con_ax.plot(test_x, obj.constraint_probability(test_model_inputs).detach())

        for z in [0, 1, 2]:
            val_ax.fill_between(
                test_x.ravel(),
                (test_mean - z * test_sigma).ravel(),
                (test_mean + z * test_sigma).ravel(),
                lw=lw,
                color=color,
                alpha=0.5**z,
            )

        ymin, ymax = val_ax.get_ylim()

        val_ax.fill_between(
            test_x.ravel(), y1=np.maximum(obj.target[0], ymin), y2=np.minimum(obj.target[1], ymax), alpha=0.2
        )
        val_ax.set_ylim(ymin, ymax)

        con_ax.set_ylabel(r"P(constraint)")
        val_ax.set_ylabel(obj.label_with_units)

        for ax in [val_ax, con_ax]:
            ax.set_xlim(*x_dof.search_domain)
            ax.set_xlabel(x_dof.label_with_units)


def _plot_objs_many_dofs(agent, axes=(0, 1), shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=32, grid_zoom=1):
    """
    Axes represents which active, non-read-only axes to plot with
    """

    plottable_dofs = agent.dofs(active=True, read_only=False)

    if gridded is None:
        gridded = len(plottable_dofs) == 2

    agent.obj_fig, agent.obj_axes = plt.subplots(
        len(agent.objectives),
        4,
        figsize=(12, 4 * len(agent.objectives)),
        constrained_layout=True,
        dpi=160,
    )

    agent.obj_axes = np.atleast_2d(agent.obj_axes)

    x_dof, y_dof = plottable_dofs[axes[0]], plottable_dofs[axes[1]]

    x_values = agent.table.loc[:, x_dof.device.name].values
    y_values = agent.table.loc[:, y_dof.device.name].values

    # test_inputs has shape (*input_shape, 1, n_active_dofs)
    # test_x and test_y should be squeezeable
    test_inputs = agent.sample(method="grid") if gridded else agent.sample(n=1024)
    test_x = test_inputs[..., 0, axes[0]].detach().squeeze().numpy()
    test_y = test_inputs[..., 0, axes[1]].detach().squeeze().numpy()

    model_inputs = agent.dofs(active=True).transform(test_inputs)

    for obj_index, obj in enumerate(agent.objectives):
        targets = agent.train_targets()[obj.name].numpy()

        values = obj._untransform(targets)

        val_vmin, val_vmax = np.nanquantile(values, q=[0.01, 0.99])
        val_norm = (
            mpl.colors.LogNorm(val_vmin, val_vmax) if obj.transform == "log" else mpl.colors.Normalize(val_vmin, val_vmax)
        )

        obj_vmin, obj_vmax = np.nanquantile(targets, q=[0.01, 0.99])
        obj_norm = mpl.colors.Normalize(obj_vmin, obj_vmax)

        val_ax = agent.obj_axes[obj_index, 0].scatter(x_values, y_values, c=values, s=size, norm=val_norm, cmap=cmap)

        # mean and sigma will have shape (*input_shape,)
        test_posterior = obj.model.posterior(model_inputs)
        test_mean = test_posterior.mean[..., 0, 0].detach().squeeze().numpy()
        test_sigma = test_posterior.variance.sqrt()[..., 0, 0].detach().squeeze().numpy()

        # test_values = obj.fitness_inverse(test_mean) if obj.kind == "fitness" else test_mean

        test_constraint = None
        if obj.constraint is not None:
            test_constraint = obj.constraint_probability(model_inputs).detach().squeeze().numpy()

        if gridded:
            # _ = agent.obj_axes[obj_index, 1].pcolormesh(
            #     test_x,
            #     test_y,
            #     test_values,
            #     shading=shading,
            #     cmap=cmap,
            #     norm=val_norm,
            # )
            if obj.constraint is not None:
                fitness_ax = agent.obj_axes[obj_index, 1].pcolormesh(
                    test_x,
                    test_y,
                    test_mean,
                    shading=shading,
                    cmap=cmap,
                    norm=obj_norm,
                )
                fit_err_ax = agent.obj_axes[obj_index, 2].pcolormesh(
                    test_x,
                    test_y,
                    test_sigma,
                    shading=shading,
                    cmap=cmap,
                    norm=mpl.colors.LogNorm(),
                )

            if test_constraint is not None:
                constraint_ax = agent.obj_axes[obj_index, 3].pcolormesh(
                    test_x,
                    test_y,
                    test_constraint,
                    shading=shading,
                    cmap=cmap,
                    # norm=mpl.colors.LogNorm(),
                )

        else:
            # _ = agent.obj_axes[obj_index, 1].scatter(
            #     test_x,
            #     test_y,
            #     c=test_values,
            #     s=size,
            #     norm=val_norm,
            #     cmap=cmap,
            # )
            if obj.constraint is not None:
                fitness_ax = agent.obj_axes[obj_index, 1].scatter(
                    test_x,
                    test_y,
                    c=test_mean,
                    s=size,
                    norm=obj_norm,
                    cmap=cmap,
                )
                fit_err_ax = agent.obj_axes[obj_index, 2].scatter(
                    test_x,
                    test_y,
                    c=test_sigma,
                    s=size,
                    cmap=cmap,
                    norm=mpl.colors.LogNorm(),
                )

            if test_constraint is not None:
                constraint_ax = agent.obj_axes[obj_index, 3].scatter(
                    test_x,
                    test_y,
                    c=test_constraint,
                    s=size,
                    cmap=cmap,
                    norm=mpl.colors.LogNorm(),
                )

        val_cbar = agent.obj_fig.colorbar(val_ax, ax=agent.obj_axes[obj_index, 0], location="bottom", aspect=32, shrink=0.8)
        val_cbar.set_label(f"{obj.units or ''}")

        if obj.constraint is not None:
            _ = agent.obj_fig.colorbar(fitness_ax, ax=agent.obj_axes[obj_index, 1], location="bottom", aspect=32, shrink=0.8)
            _ = agent.obj_fig.colorbar(fit_err_ax, ax=agent.obj_axes[obj_index, 2], location="bottom", aspect=32, shrink=0.8)

            # obj_cbar.set_label(f"{obj.label}")
            # err_cbar.set_label(f"{obj.label}")

        if test_constraint is not None:
            constraint_cbar = agent.obj_fig.colorbar(
                constraint_ax, ax=agent.obj_axes[obj_index, 3], location="bottom", aspect=32, shrink=0.8
            )

            constraint_cbar.set_label(f"{obj.label_with_units} constraint")

        col_names = [
            f"{obj.description} samples",
            f"pred. {obj.description} fitness",
            f"pred. {obj.description} fitness error",
            f"{obj.description} constraint",
        ]

        pad = 5

        for column_index, ax in enumerate(agent.obj_axes[obj_index]):
            ax.set_title(
                col_names[column_index],
            )

    if len(agent.objectives) > 1:
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
        ax.set_xlabel(x_dof.label_with_units)
        ax.set_ylabel(y_dof.label_with_units)
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
    agent, acqfs, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, gridded=None, size=16, **kwargs
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
    x_values = agent.table.loc[:, x_dof.device.name].values

    test_inputs = agent.sample(method="grid")
    constraint = agent.constraint(agent.dofs.transform(test_inputs))[..., 0]

    agent.valid_ax.scatter(x_values, agent.all_objectives_valid, s=size)
    agent.valid_ax.plot(test_inputs.squeeze(-2), constraint, lw=lw)
    agent.valid_ax.set_xlim(*x_dof.search_domain)


def _plot_valid_many_dofs(agent, axes=[0, 1], shading="nearest", cmap=DEFAULT_COLORMAP, size=16, gridded=None):
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
    x = getattr(agent.table, x_key).values

    num_obj_plots = 1
    if show_all_objs:
        num_obj_plots = len(agent.objectives) + 1

    len(agent.objectives) + 1 if len(agent.objectives) > 1 else 1

    hist_fig, hist_axes = plt.subplots(
        num_obj_plots, 1, figsize=(6, 4 * num_obj_plots), sharex=True, constrained_layout=True, dpi=200
    )
    hist_axes = np.atleast_1d(hist_axes)

    unique_strategies, acqf_index, acqf_inverse = np.unique(agent.table.acqf, return_index=True, return_inverse=True)

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

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    y = agent.train_targets(fitness=True, concatenate=True)

    pareto_mask = agent.pareto_mask
    constraint = agent.evaluated_constraints.all(axis=-1)

    ax.scatter(*y[(~pareto_mask) & constraint].T[[i, j]], c="k")
    ax.scatter(*y[~constraint].T[[i, j]], c="r", marker="x", label="invalid")
    ax.scatter(*y[pareto_mask].T[[i, j]], c="b", label="Pareto front")

    ax.set_xlabel(f"{f_objs[i].name} fitness")
    ax.set_ylabel(f"{f_objs[j].name} fitness")

    ax.legend()
