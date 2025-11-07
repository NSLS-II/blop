---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3.11.5 64-bit
  language: python
  name: python3
---

# Finding latent dimensions for the toroidal mirror 

It is common that beamline inputs are highly coupled, and so the effect of an input on the beam cannot be understood except in concert with the others. In this example, we show how our agent figures out latent dimensions, as well as the benefit of doing so. 

```{code-cell} ipython3
from blop.utils import prepare_re_env  # noqa: F401

%run -i $prepare_re_env.__file__ --db-type=temp
%run -i ../../../examples/prepare_tes_shadow.py
```

```{code-cell} ipython3
from blop.experiments.sirepo.tes import w8_digestion

import blop

dofs = [
    {"device": toroid.x_rot, "limits": (-0.001, 0.001), "kind": "active"},
    {"device": toroid.offz, "limits": (-0.5, 0.5), "kind": "active"},
]

tasks = [{"key": "flux", "kind": "maximize", "transform": "log"}]

agent = blop.bayesian.Agent(
    dofs=dofs,
    tasks=tasks,
    dets=[w8],
    digestion=w8_digestion,
    db=db,
)

RE(agent.initialize("qr", n_init=24))
```

We can see that the beam is only not cut off (i.e. it has a non-zero flux) in a diagonal strip, and that in fact this is really just a one-dimensional optimization problem in some diagonal dimension. Our agent has figured this out, with a transformation matrix that has a long coherence length in one dimension and a short coherence length orthogonal to it:

```{code-cell} ipython3
agent.tasks[0]["model"].covar_module.latent_transform
```

```{code-cell} ipython3
agent.plot_objectives()
agent.plot_constraint()
agent.plot_acquisition(strategy=["ei", "pi", "ucb"])
```
