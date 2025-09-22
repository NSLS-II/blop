---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3.11.4 64-bit
  language: python
  name: python3
---

# Multi-task optimization of KB mirrors

Often, we want to optimize multiple aspects of a system; in this real-world example aligning the Kirkpatrick-Baez mirrors at the TES beamline's endstation, we care about the horizontal and vertical beam size, as well as the flux. 

We could try to model these as a single task by combining them into a single number (i.e., optimization the beam density as flux divided by area), but our model then loses all information about how different inputs affect different outputs. We instead give the optimizer multiple "tasks", and then direct it based on its prediction of those tasks. 

```{code-cell} ipython3
from blop.utils import prepare_re_env  # noqa: F401

%run -i $prepare_re_env.__file__ --db-type=temp
%run -i ../../../examples/prepare_tes_shadow.py
```

```{code-cell} ipython3
from blop.experiments.sirepo.tes import w9_digestion

from blop.bayesian import Agent

dofs = [
    {"device": kbv.x_rot, "limits": (-0.1, 0.1), "kind": "active"},
    {"device": kbh.x_rot, "limits": (-0.1, 0.1), "kind": "active"},
]

tasks = [
    {"key": "flux", "kind": "maximize", "transform": "log"},
    {"key": "w9_fwhm_x", "kind": "minimize", "transform": "log"},
    {"key": "w9_fwhm_y", "kind": "minimize", "transform": "log"},
]

agent = Agent(
    dofs=dofs,
    tasks=tasks,
    dets=[w9],
    digestion=w9_digestion,
    db=db,
)

RE(agent.initialize("qr", n_init=4))
```

```{code-cell} ipython3
RE(agent.learn("ei"))
```

For each task, we plot the sampled data and the model's posterior with respect to two inputs to the KB mirrors. We can see that each tasks responds very differently to different motors, which is very useful to the optimizer. 

```{code-cell} ipython3
agent.plot_objectives()
agent.plot_acqfuisition(strategy=["ei", "pi", "ucb"])
```

We should find our optimum (or something close to it) on the very next iteration:

```{code-cell} ipython3
RE(agent.learn("ei", n_iter=2))
agent.plot_objectives()
```

The agent has learned that certain dimensions affect different tasks differently!
