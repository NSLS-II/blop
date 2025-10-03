---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Custom acquisition plans

The simplest acqusition plan for a beamline is to move some motor inputs to some positions, and then trigger some detectors. Often, though, we want to investigate behaviors more complex than this. Consider the example of aligning the spectrometer at the Inner-Shell Spectroscopy (ISS) beamline at NSLS-II, which operates by modulating the energy of a beam on a sample and watching the resulting flux rise, peak, and then fall. 

We can build a toy model of this spectrometer using custom acquisition and digestion functions. Let's pretend that we can vary two inputs $\mathbf{x} = (x_1, x_2)$, and that the resolution of the spectrometer is equal to 

$\nu_\sigma = \big (1 + x_1^2 + (x_2 - 1)^2 \big)^{1/2}$

which has a single minimum at $(0,1)$. We can't sample this value directly, rather we can sample how the resulting flux peaks as we vary the energy $\nu$. Let's pretend that it looks like a Gaussian:

$I(\nu) = \exp \Big [ - 0.5 (\nu - \nu_0)^2 / \nu_\sigma^2 \Big ]$

To find the inputs that lead to the tightest spectrum, we need to vary $\mathbf{x}$, scan over $\nu$, and then estimate the resolution for the agent to optimize over. Let's write acquisition and digestion functions to do this: 

```{code-cell} ipython3
import numpy as np


def acquisition(dofs, inputs, dets):
    _ = yield from bp.list_scan

    for x in inputs:
        _ = np.sqrt(1 + np.square(x).sum())  # our resolution is

        nu_sigma = np.sqrt(1 + x1**2 + (x2 - 1) ** 2)

        _ = np.exp(-0.5 * np.square((nu - nu_0) / nu_sigma))
```

```{code-cell} ipython3
from matplotlib import pyplot as plt

nu_0 = 100

nu = np.linspace(90, 110, 256)

for x1, x2 in [(0, 0), (-2, 2), (-1, 0), (0, 1)]:
    nu_sigma = np.sqrt(1 + x1**2 + (x2 - 1) ** 2)

    flux = np.exp(-0.5 * np.square((nu - nu_0) / nu_sigma))

    plt.plot(nu, flux, label=f"(x1, x2) = ({x1}, {x2})")

plt.legend()
```

To find the inputs that lead to the tightest spectrum, we need to vary $\mathbf{x}$, scan over $\nu$, and then estimate the resolution for the agent to optimize over. Let's write acquisition and digestion functions to do this:
