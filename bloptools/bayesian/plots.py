import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLOR_LIST = ["dodgerblue", "tomato", "mediumseagreen", "goldenrod"]
DEFAULT_COLORMAP = "turbo"
DEFAULT_SCATTER_SIZE = 16

class TasksPlotManyDOFs:

    def __init__(self, agent, **kwargs):

        self.cmap = kwargs.get("cmap", DEFAULT_COLORMAP)
        self.s = kwargs.get("s", DEFAULT_SCATTER_SIZE)
        self.gridded = kwargs.get("gridded", False)
        self.axes = kwargs.get("axes", [0, 1])

        self.make_plots(agent)

    def make_plots(self, agent):

        self.task_fig, self.task_axes = plt.subplots(
            len(agent.tasks),
            3,
            figsize=(12, 4 * len(agent.tasks)),
            sharex=True,
            constrained_layout=True,
        )

        self.task_axes = np.atleast_2d(self.task_axes)

        for itask in range(len(agent.tasks)):

            self.task_axes[itask, 0].scatter([], [], s=self.s, cmap=self.cmap)

            if self.gridded:
                self.task_axes[itask, 1].imshow([[0]], cmap=self.cmap)
                self.task_axes[itask, 2].imshow([[0]], cmap=self.cmap)

            else:
                self.task_axes[itask, 1].scatter([], [], s=self.s, cmap=self.cmap)
                self.task_axes[itask, 2].scatter([], [], s=self.s, cmap=self.cmap)

            self.task_fig.colorbar(self.task_axes[itask, 0].collections[0], ax=self.task_axes[itask, :2], location="bottom", aspect=32, shrink=0.8)
            self.task_fig.colorbar(self.task_axes[itask, 2].collections[0], ax=self.task_axes[itask, 2], location="bottom", aspect=32, shrink=0.8)


    def update(self, agent):

        for itask, task in enumerate(agent.tasks):

            sampled_fitness = agent.table.loc[:, f'{task["key"]}_fitness'].values
            task_vmin, task_vmax = np.nanpercentile(sampled_fitness, q=[1, 99])
            task_norm = mpl.colors.Normalize(task_vmin, task_vmax)

            data_scatter = self.task_axes[itask, 0].collections[0]
            pred_mean    = self.task_axes[itask, 1].collections[0]
            pred_error   = self.task_axes[itask, 2].collections[0]

            data_scatter.set_offsets(agent.inputs.values[:, self.axes])
            data_scatter.set_array(sampled_fitness)
            data_scatter.set_norm(task_norm)

            x = agent.test_inputs_grid if self.gridded else agent.test_inputs(n=1024).squeeze()

            task_posterior = task["model"].posterior(x)
            task_pred_mean = task_posterior.mean.squeeze()
            task_pred_sigma = task_posterior.variance.sqrt().squeeze() 

            if self.gridded:
                if not x.ndim == 3:
                    raise ValueError()
                pred_mean.set_array(task_pred_mean.detach())
                pred_error.set_array(task_pred_sigma.detach())

            else:
                pred_mean.set_offsets(x.detach()[..., self.axes])
                pred_error.set_offsets(x.detach()[..., self.axes])
                pred_mean.set_array(task_pred_mean.detach())
                pred_error.set_array(task_pred_sigma.detach())

        for ax in self.task_axes.ravel():
            ax.set_xlim(*agent._subset_dofs(kind="active", mode="on")[self.axes[0]]["limits"])
            ax.set_ylim(*agent._subset_dofs(kind="active", mode="on")[self.axes[1]]["limits"])
