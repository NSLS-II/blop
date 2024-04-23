import asyncio

import databroker
import matplotlib as mpl
import numpy as np
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from nicegui import ui

from blop import DOF, Agent, Objective
from blop.utils import functions

# MongoDB backend:
db = Broker.named("temp")  # mongodb backend
try:
    databroker.assets.utils.install_sentinels(db.reg.config, version=1)
except Exception:
    pass

loop = asyncio.new_event_loop()
loop.set_debug(True)
RE = RunEngine({}, loop=loop)
RE.subscribe(db.insert)

bec = best_effort.BestEffortCallback()
RE.subscribe(bec)

bec.disable_baseline()
bec.disable_heading()
bec.disable_table()
bec.disable_plots()


dofs = [
    DOF(name="x1", description="x1", search_domain=(-5.0, 5.0)),
    DOF(name="x2", description="x2", search_domain=(-5.0, 5.0)),
]

objectives = [Objective(name="himmelblau", target="min")]

agent = Agent(
    dofs=dofs,
    objectives=objectives,
    digestion=functions.himmelblau_digestion,
    db=db,
    verbose=True,
    tolerate_acquisition_errors=False,
)

agent.acqf_index = 0

agent.acqf_number = 2


with ui.pyplot(figsize=(10, 4), dpi=160) as obj_plt:
    extent = [*agent.dofs[0].search_domain, *agent.dofs[1].search_domain]

    ax1 = obj_plt.fig.add_subplot(131)
    ax1.set_title("Samples")
    im1 = ax1.scatter([], [], cmap="magma")

    ax2 = obj_plt.fig.add_subplot(132, sharex=ax1, sharey=ax1)
    ax2.set_title("Posterior mean")
    im2 = ax2.imshow(np.random.standard_normal(size=(32, 32)), extent=extent, cmap="magma")

    ax3 = obj_plt.fig.add_subplot(133, sharex=ax1, sharey=ax1)
    ax3.set_title("Posterior error")
    im3 = ax3.imshow(np.random.standard_normal(size=(32, 32)), extent=extent, cmap="magma")

    data_cbar = obj_plt.fig.colorbar(mappable=im1, ax=[ax1, ax2], location="bottom", aspect=32)
    err_cbar = obj_plt.fig.colorbar(mappable=im3, ax=[ax3], location="bottom", aspect=16)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(agent.dofs[0].label_with_units)
        ax.set_ylabel(agent.dofs[1].label_with_units)


acqf_configs = {
    0: {"name": "qr", "long_name": r"quasi-random sampling"},
    1: {"name": "qei", "long_name": r"$q$-expected improvement"},
    2: {"name": "qpi", "long_name": r"$q$-probability of improvement"},
    3: {"name": "qucb", "long_name": r"$q$-upper confidence bound"},
}

with ui.pyplot(figsize=(10, 3), dpi=160) as acq_plt:
    extent = [*agent.dofs[0].search_domain, *agent.dofs[1].search_domain]

    acqf_plt_objs = {}

    for iax, config in acqf_configs.items():
        if iax == 0:
            continue

        acqf = config["name"]

        acqf_plt_objs[acqf] = {}

        acqf_plt_objs[acqf]["ax"] = ax = acq_plt.fig.add_subplot(1, len(acqf_configs) - 1, iax)

        ax.set_title(config["long_name"])
        acqf_plt_objs[acqf]["im"] = ax.imshow([[]], extent=extent, cmap="gray_r")
        acqf_plt_objs[acqf]["hist"] = ax.scatter([], [])
        acqf_plt_objs[acqf]["best"] = ax.scatter([], [])

        ax.set_xlabel(agent.dofs[0].label_with_units)
        ax.set_ylabel(agent.dofs[1].label_with_units)


acqf_button_options = {index: config["name"] for index, config in acqf_configs.items()}

v = ui.checkbox("visible", value=True)
with ui.column().bind_visibility_from(v, "value"):
    ui.toggle(acqf_button_options).bind_value(agent, "acqf_index")
    ui.number().bind_value(agent, "acqf_number")


def reset():
    agent.reset()

    print(agent.table)


def learn():
    acqf_config = acqf_configs[agent.acqf_index]

    acqf = acqf_config["name"]

    n = int(agent.acqf_number) if acqf != "qr" else 16

    ui.notify(f"sampling {n} points with acquisition function \"{acqf_config['long_name']}\"")

    RE(agent.learn(acqf, n=n))

    with obj_plt:
        obj = agent.objectives[0]

        x_samples = agent.raw_inputs().detach().numpy()
        y_samples = agent.raw_targets(obj.name).detach().numpy()[..., 0]

        x = agent.sample(method="grid", n=20000)  # (n, n, 1, d)
        model_x = agent.dofs.transform(x)
        p = obj.model.posterior(model_x)

        m = p.mean.squeeze(-1, -2).detach().numpy()
        e = p.variance.sqrt().squeeze(-1, -2).detach().numpy()

        im1.set_offsets(x_samples)
        im1.set_array(y_samples)
        im1.set_cmap("magma")

        im2.set_data(m.T[::-1])
        im3.set_data(e.T[::-1])

        obj_norm = mpl.colors.Normalize(vmin=np.nanmin(y_samples), vmax=np.nanmax(y_samples))
        err_norm = mpl.colors.LogNorm(vmin=np.nanmin(e), vmax=np.nanmax(e))

        im1.set_norm(obj_norm)
        im2.set_norm(obj_norm)
        im3.set_norm(err_norm)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(*agent.dofs[0].search_domain)
            ax.set_ylim(*agent.dofs[1].search_domain)

    with acq_plt:
        x = agent.sample(method="grid", n=20000)  # (n, n, 1, d)
        model_x = agent.dofs.transform(x)
        x_samples = agent.train_inputs().detach().numpy()

        for acqf in acqf_plt_objs.keys():
            ax = acqf_plt_objs[acqf]["ax"]

            acqf_obj = getattr(agent, acqf)(model_x).detach().numpy()

            acqf_norm = mpl.colors.Normalize(vmin=np.nanmin(acqf_obj), vmax=np.nanmax(acqf_obj))
            acqf_plt_objs[acqf]["im"].set_data(acqf_obj.T[::-1])
            acqf_plt_objs[acqf]["im"].set_norm(acqf_norm)

            res = agent.ask(acqf, n=int(agent.acqf_number))

            acqf_plt_objs[acqf]["hist"].remove()
            acqf_plt_objs[acqf]["hist"] = ax.scatter(*x_samples.T, ec="b", fc="none", marker="o")

            acqf_plt_objs[acqf]["best"].remove()
            acqf_plt_objs[acqf]["best"] = ax.scatter(*res["points"].T, c="r", marker="x", s=64)

            ax.set_xlim(*agent.dofs[0].search_domain)
            ax.set_ylim(*agent.dofs[1].search_domain)


ui.button("Learn", on_click=learn)

ui.button("Reset", on_click=reset)

ui.run(port=8004)
