Objectives
++++++++++

Objectives are what control how optimal the output of our experiment is, and are defined by ``Objective`` objects.

``blop`` combines one or many ``Objective`` objects into an ``ObjectiveList``, which encapsulates how we model and optimize our outputs.

Fitness
-------

A fitness objective is an ``Objective`` that minimizes or maximizes a given value.

* Maximize the flux of a beam of light.
* Minimize the size of a beam.

We can construct an objective to maximize some output with

.. code-block:: python

    from blop import Objective

    objective = Objective(name="some_output", target="max") # or "min"

Given some data, the ``Objective`` object will try to model the quantity ``some_output`` and find the corresponding inputs that maximize it.
We can also apply a transform to the value to make it more Gaussian when we fit to it.
This is especially useful when the quantity tends to be non-Gaussian, like with a beam flux.

.. code-block:: python

    from blop import Objective

    objective_with_log_transform = Objective(name="some_output", target="max", transform="log")

    objective_with_arctanh_transform = Objective(name="some_output", target="max", transform="arctanh")


.. code-block:: python

    objective = Objective(name="y1", target=(1, 3)) # find any input that puts "y1" between 1 and 3.


Often, the objective being modeled is some strictly positive quantity (e.g. the size of a beam being aligned).
In this case, a smart thing to do is to set ``log=True``, which will model and subsequently optimize the logarithm of the objective:

.. code-block:: python

    from blop import Objective

    objective = Objective(name="some_strictly_positive_output", target="max", log=True)


Constraints
-----------

A constraint objective doesn't try to minimize or maximize a value, and instead just tries to maximize the chance that the objective is within some acceptable range.
This is useful in optimization problems like

* Require a beam to be within some box.
* Require the wavelength of light to be a certain color.
* We want a beam to be focused enough to perform some experiment, but not necessarily optimal.

.. code-block:: python

    # ensure the color is approximately green
    color_bjective = Objective(name="peak_wavelength", target=(520, 530), units="nm")

    # ensure the beam is smaller than 10 microns
    width_objective = Objective(name="beam_width", target=(-np.inf, 10), units="um", transform="log")

    # ensure our flux is at least some value
    flux_objective = Objective(name="beam_flux", target=(1.0, np.inf), transform="log")



Validity
--------

A common problem in beamline optimization is in the random or systematically invalid measurement of objectives. This arises in different ways, like when

* The beam misses the detector, leading our beam parser to return some absurdly small or large value.
* Some part of the experiment glitches, leading to an uncharacteristic data point.
* Some part of the data postprocessing pipeline fails, giving no value for the output.

We obviously want to exclude these points from our model fitting, but if we stopped there, inputs that always lead to invalid outputs will lead to an infinite loop of trying to sample an interesting but invalid points (as the points get immediately removed every time).
The set of points that border valid and invalid data points are often highly nonlinear and unknown *a priori*.
We solve this by implementing a validity model for each ``Objective``, which constructs and fits a probabilistic model for validity for all inputs.
Using this model, we constrain acquisition functions to take into account the possibility that the output value is invalid, meaning it will eventually learn to ignore infeasible points.

We can control the exclusion of data points in two ways. The first is to specify a ``trust_domain`` for the objective, so that the model only "trusts" points in that domain. For example:

.. code-block:: python

    # any beam smaller than two pixels shouldn't be trusted.
    # any beam larger than 100 pixels will mess up our model and aren't interesting anyway
    objective = Objective(name="beam_size", trust_domain=(2, 100), units="pixels")

This will set any value outside of the ``trust_domain`` to ``NaN``, which the model will learn to avoid.
The second way is to ensure that any invalid values are converted to ``NaN`` in the diagnostics, before the agent ever sees them.
