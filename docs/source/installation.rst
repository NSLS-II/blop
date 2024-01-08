=========================
Installation instructions
=========================

Installation
------------

The package works with Python 3.8+ and can be installed from both PyPI and conda-forge.

To install the package using the ``pip`` package manager, run the following command:

.. code:: bash

   $ python3 -m pip install blop

To install the package using the ``conda`` package manager, run the following command:

.. code:: bash

   $ conda install -c conda-forge blop

If you'd like to use the Sirepo backend and ``sirepo-bluesky`` ophyd objects, please
follow the `Sirepo/Sirepo-Bluesky installation & configuration instructions
<https://nsls-ii.github.io/sirepo-bluesky/installation.html>`_.


Run tests
---------

.. code:: bash

   $ pytest -vv -s -x --pdb


Build documentation
-------------------

.. code:: bash

   $ make -C docs/ html
