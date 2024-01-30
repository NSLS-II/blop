Objectives
++++++++++

We can describe an optimization problem as a list of objectives to. A simple objective is

.. code-block:: python

    from blop import Objective

    objective = Objective(name="y1", target="max")

Given some data, the ``Objective`` object will try to model the quantity "y1" and find the corresponding inputs that maximize it.
The objective will expect that this quantity will be spit out by the experimentation loop, so we will check later that it is set up correctly.
There are many ways to specify an objective's behavior, which is done by changing the objective's target:

.. code-block:: python

    from blop import Objective

    objective = Objective(name="y1", target="min") # minimize the quantity "y1"

    objective = Objective(name="y1", target=2) # get the quantity "y1" as close to 2 as possible

    objective = Objective(name="y1", target=(1, 3)) # find any input that puts "y1" between 1 and 3.


Often, the objective being modeled is some strictly positive quantity (e.g. the size of a beam being aligned).
In this case, a smart thing to do is to set ``log=True``, which will model and subsequently optimize the logarithm of the objective:

.. code-block:: python

    from blop import Objective

    objective = Objective(name="some_strictly_positive_quantity", target="max", log=True)
