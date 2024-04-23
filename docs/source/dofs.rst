Degrees of freedom (DOFs)
+++++++++++++++++++++++++

Continuous degrees of freedom
-----------------------------

A degree of freedom is a variable that affects our optimization objective. We can define a simple DOF as

.. code-block:: python

    from blop import DOF

    dof = DOF(name="x1", description="my first DOF", search_domain=(lower, upper))

This will instantiate a bunch of stuff under the hood, so that our agent knows how to move things and where to search.
Typically, this will correspond to a real, physical device available in Python. In that case, we can pass the DOF an ophyd device in place of a name

.. code-block:: python

    from blop import DOF

    dof = DOF(device=my_ophyd_device, description="a real piece of hardware", search_domain=(lower, upper))

In this case, the agent will control the device as it sees fit, moving it between the search bounds.

Sometimes, a DOF may be something we can't directly control (e.g. a changing synchrotron current or a changing sample temperature) but want our agent to be aware of.
In this case, we can define a read-only DOF as

.. code-block:: python

    from blop import DOF

    dof = DOF(device=a_read_only_ophyd_device, description="a thermometer or something", read_only=True, trust_domain=(lower, upper))

and the agent will use the received values to model its objective, but won't try to move it.
We can also pass a set of ``trust_domain``, so that our agent will ignore experiments where the DOF value jumps outside of the interval.


Discrete degrees of freedom
---------------------------

In addition to degrees of freedom that vary continuously between a lower and upper bound, we can define discrete degrees of freedom.
One kind is a binary degree of freedom, where the input can take one of two values, e.g.

.. code-block:: python

    discrete_dof = DOF(name="x1", description="A discrete DOF", type="discrete", search_domain={"in", "out"})

Another is an ordinal degree of freedom, which takes more than two discrete values but has some ordering, e.g.

.. code-block:: python

    ordinal_dof = DOF(name="x1", description="An ordinal DOF", type="ordinal", search_domain={"low", "medium", "high"})

The last is a categorical degree of freedom, which can take many different discrete values with no ordering, e.g.

.. code-block:: python

    categorical_dof = DOF(name="x1", description="A categorical DOF", type="categorical", search_domain={"banana", "mango", "papaya"})
