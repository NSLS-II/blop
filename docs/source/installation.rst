=========================
Installation instructions
=========================

Installation
------------

The package works with Python 3.10+ and can be installed from both PyPI and conda-forge.

To install the package using the ``pip`` package manager, run the following command:

.. code:: bash

   $ python -m pip install blop

To install the package using the ``conda`` package manager, run the following command:

.. code:: bash

   $ conda install -c conda-forge blop

If you'd like to use the Sirepo backend and ``sirepo-bluesky`` ophyd objects, please
follow the `Sirepo/Sirepo-Bluesky installation & configuration instructions
<https://nsls-ii.github.io/sirepo-bluesky/installation.html>`_.

For Development
---------------

Install Pixi
^^^^^^^^^^^^

We use Pixi to manage our development environment. Go to https://pixi.sh/latest/installation/ to install it.

Run tests
^^^^^^^^^

.. code:: bash

   $ pixi run unit-tests


Build documentation
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   $ pixi run build-docs
