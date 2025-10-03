---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3.11.4 ('bluesky')
  language: python
  name: python3
---

# Introduction (Himmelblau's function)


+++

Let's use ``blop`` to minimize Himmelblau's function, subject to the constraint that $x_1^2 + x_2^2 < 50$. Our function looks like this:

```{code-cell} ipython3
import matplotlib as mpl
import numpy as np
from blop.tasks import Task
from matplotlib import pyplot as plt

from blop.utils import functions

x1 = x2 = np.linspace(-8, 8, 256)
X1, X2 = np.meshgrid(x1, x2)

task = Task(name="himmelblau", kind="min")
F = functions.constrained_himmelblau(X1, X2)

plt.pcolormesh(x1, x2, F, norm=mpl.colors.LogNorm(), cmap="gnuplot")
plt.colorbar()
plt.xlabel("x1")
plt.ylabel("x2")
```

There are several things that our agent will need. The first ingredient is some degrees of freedom (these are always `ophyd` devices) which the agent will move around to different inputs within each DOF's bounds (the second ingredient). We define these here:

```{code-cell} ipython3
from blop import devices

dofs = [
    {"device": devices.DOF(name="x1"), "limits": (-8, 8), "kind": "active"},
    {"device": devices.DOF(name="x2"), "limits": (-8, 8), "kind": "active"},
]
```

We also need to give the agent something to do. We want our agent to look in the feedback for a variable called "himmelblau", and try to minimize it.

```{code-cell} ipython3
tasks = [
    {"key": "himmelblau", "kind": "minimize"},
]
```

In our digestion function, we define our objective as a deterministic function of the inputs, returning a `NaN` when we violate the constraint:

```{code-cell} ipython3
def digestion(db, uid):
    products = db[uid].table()

    for index, entry in products.iterrows():
        products.loc[index, "himmelblau"] = functions.constrained_himmelblau(entry.x1, entry.x2)

    return products
```

We then combine these ingredients into an agent, giving it an instance of ``databroker`` so that it can see the output of the plans it runs.

```{code-cell} ipython3
from blop.utils import prepare_re_env  # noqa: F401

%run -i $prepare_re_env.__file__ --db-type=temp
from blop.bayesian import Agent

agent = Agent(
    dofs=dofs,
    tasks=tasks,
    digestion=digestion,
    db=db,
)
```

```{code-cell} ipython3
import blop

blop.bayesian.acquisition.parse_acq_func_identifier(acq_func_identifier="quasi-random")
```

Without any data, we can't make any inferences about what the function looks like, and so we can't use any non-trivial acquisition functions. 

```{code-cell} ipython3
RE(agent.learn("quasi-random", n=64))
agent.plot_objectives()
```

In addition to modeling the fitness of the task, the agent models the probability that an input will be feasible:

```{code-cell} ipython3
agent.plot_constraint(cmap="viridis")
```

It combines the estimate of the objective and the estimate of the feasibility in deciding where to go:

```{code-cell} ipython3
X = agent.ask("qei", n=8)
```

```{code-cell} ipython3
import scipy as sp

X = sp.interpolate.interp1d(np.arange(len(X)), X, axis=0, kind="cubic")(np.linspace(0, len(X) - 1, 16))
plt.plot(*X.T)
```

```{code-cell} ipython3
agent.plot_acquisition(acq_func=["ei", "pi", "ucb"], cmap="viridis")
plt.plot(*X.T, c="r", marker="x")
```

```{code-cell} ipython3
import yaml

with open("config.yml", "w") as f:
    yaml.safe_dump(ACQ_FUNC_CONFIG, f)
```

```{code-cell} ipython3
RE(agent.learn("qei", n_per_iter=4))
```

The agent automatically tries to avoid infeasible points, but will end up naturally exploring the boundary of the constraint. Let's see where the agent is thinking of going:

```{code-cell} ipython3
agent.plot_objectives()
agent.plot_acquisition(strategy=["ei", "pi", "ucb"])
```

The agent will naturally explore the whole parameter space

```{code-cell} ipython3
RE(agent.learn("ei", n_iter=16))
agent.plot_objectives()
```
