---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3.10.12 ('bluesky')
  language: python
  name: python3
---

# Bayesian optimization

+++

This tutorial is an introduction to the syntax used by the optimizer, as well as the principles of Bayesian optimization in general.

We'll start by minimizing the Styblinski-Tang function in one dimension, which looks like this:

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt

from blop.utils import functions

x = np.linspace(-5, 5, 256)

plt.plot(x, functions.styblinski_tang(x), c="b")
plt.xlim(-5, 5)
```

There are several things that our agent will need. The first ingredient is some degrees of freedom (these are always `ophyd` devices) which the agent will move around to different inputs within each DOF's bounds (the second ingredient). We define these here:

```{code-cell} ipython3
from blop import devices

dofs = [
    {"device": devices.DOF(name="x"), "limits": (-5, 5), "kind": "active"},
]
```

```{code-cell} ipython3
tasks = [
    {"key": "styblinski-tang", "kind": "minimize"},
]
```


This degree of freedom will move around a variable called `x1`. The agent automatically samples at different inputs, but we often need some post-processing after data collection. In this case, we need to give the agent a way to compute the Styblinski-Tang function. We accomplish this with a digestion function, which always takes `(db, uid)` as an input. For each entry, we compute the function:

```{code-cell} ipython3
def digestion(db, uid):
    products = db[uid].table()

    for index, entry in products.iterrows():
        products.loc[index, "styblinski-tang"] = functions.styblinski_tang(entry.x)

    return products
```

The next ingredient is a task, which gives the agent something to do. We want it to minimize the Styblinski-Tang function, so we make a task that will try to minimize the output of the digestion function called "styblinski-tang".

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

RE(agent.initialize("qr", n_init=4))
```

We initialized the GP with the "quasi-random" strategy, as it doesn't require any prior data. We can view the state of the optimizer's posterior of the tasks over the input parameters:

```{code-cell} ipython3
# what are the points?

agent.plot_objectives()
```

Note that the value of the fitness is the negative value of the function: we always want to maximize the fitness of the tasks.

An important concept in Bayesian optimization is the acquisition function, which is how the agent decides where to sample next. Under the hood, the agent will see what inputs maximize the acquisition function to make its decision.

We can see what the agent is thinking by asking it to plot a few different acquisition functions in its current state.

```{code-cell} ipython3
agent.all_acq_funcs
```

```{code-cell} ipython3
agent.plot_acqfuisition(acq_funcs=["ei", "pi", "ucb"])
```

Let's tell the agent to learn a little bit more. We just have to tell it what acquisition function to use (by passing a `strategy`) and how many iterations we'd like it to perform (by passing `n_iter`).

```{code-cell} ipython3
RE(agent.learn("ei", n_iter=4))
agent.plot_objectives()
```
