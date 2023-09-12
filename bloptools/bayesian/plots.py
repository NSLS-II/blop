import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class TasksPlotManyDOFs:
    def __init__(self):
        self.task_fig, self.task_axes = plt.subplots(
            self.n_tasks,
            1,
            figsize=(6, 4 * self.n_tasks),
            sharex=True,
            constrained_layout=True,
        )

        self.axes = [0, 1]

        self.update()

    def update(self, dofs, tasks):
        gridded = self._len_subset_dofs(kind="active", mode="on") == 2

        for itask, task in enumerate(self.tasks):
            task_plots = self.plots["tasks"][task["name"]]

            sampled_fitness = self.table.loc[:, f'{task["key"]}_fitness'].values
            task_vmin, task_vmax = np.nanpercentile(sampled_fitness, q=[1, 99])
            task_norm = mpl.colors.Normalize(task_vmin, task_vmax)
            task_plots["sampled"].set_norm(task_norm)
            task_plots["sampled"].set_norm(task_norm)

            task_plots["sampled"].set_offsets(self.inputs.values[:, self.axes])
            task_plots["sampled"].set_array(sampled_fitness)

            x = self.test_inputs_grid.squeeze() if gridded else self.test_inputs(n=1024)

            task_posterior = task["model"].posterior(x)
            task_mean = task_posterior.mean
            task_sigma = task_posterior.variance.sqrt()

            if gridded:
                if not x.ndim == 3:
                    raise ValueError()
                task_plots["pred_mean"].set_array(task_mean[..., 0].detach().numpy())
                task_plots["pred_sigma"].set_array(task_sigma[..., 0].detach().numpy())

            else:
                task_plots["pred_mean"].set_offsets(x.detach().numpy()[..., self.axes])
                task_plots["pred_mean"].set_array(task_mean[..., 0].detach().numpy())
                task_plots["pred_sigma"].set_offsets(x.detach().numpy()[..., self.axes])
                task_plots["pred_sigma"].set_array(task_sigma[..., 0].detach().numpy())

        for ax in self.task_axes.ravel():
            ax.set_xlim(*self._subset_dofs(kind="active", mode="on")[self.axes[0]]["limits"])
            ax.set_ylim(*self._subset_dofs(kind="active", mode="on")[self.axes[1]]["limits"])
