
.. warning::

    **Important notice for the upcoming v1.0.0 release**

    Major changes are coming in the v1.0.0 release of Blop. We are removing the older way of using agents and performing optimization in favor of using `Ax <https://ax.dev>`_ as the backend for optimization and experiment tracking. The legacy agent interface is now deprecated, and users are encouraged to migrate to the new `blop.ax.Agent` interface for all optimization workflows. Please refer to the :doc:`tutorials`, :doc:`how-to-guides`, and :doc:`explanation` sections for the new interface.

    Furthermore, Blop is moving to a more modular architecture. The core architecture is now protocol-based, allowing users to plug in their own optimization backends, acquisition plans, and evaluation functions.
    
    Ax will have first-class support as the default backend for optimization and experiment tracking.


What is Blop?
-------------

Blop is a Python library for performing optimization for beamline experiments. It is designed to integrate nicely with the Bluesky ecosystem and primarily acts as a bridge between optimization routines and fine-grained data acquisition and control. Our goal is to provide a simple and practical data-driven optimization interface for beamline experiments.


Documentation structure
-----------------------

- :doc:`installation` - Installation instructions
- :doc:`how-to-guides` - How-to guides for common tasks
- :doc:`explanation` - Explanation of the underlying concepts
- :doc:`tutorials` - Tutorials for learning
- :doc:`reference` - Reference documentation for the API
- :doc:`release-history` - Release history

.. toctree::
    :maxdepth: 2
    :hidden:

    installation
    how-to-guides
    explanation
    tutorials
    reference
    release-history

Citation
--------

If you use this package in your work, please cite the following paper:

  Morris, T. W., Rakitin, M., Du, Y., Fedurin, M., Giles, A. C., Leshchev, D., Li, W. H., Romasky, B., Stavitski, E., Walter, A. L., Moeller, P., Nash, B., & Islegen-Wojdyla, A. (2024). A general Bayesian algorithm for the autonomous alignment of beamlines. Journal of Synchrotron Radiation, 31(6), 1446–1456. https://doi.org/10.1107/S1600577524008993

BibTeX:

.. code-block:: bibtex

   @Article{Morris2024,
     author   = {Morris, Thomas W. and Rakitin, Max and Du, Yonghua and Fedurin, Mikhail and Giles, Abigail C. and Leshchev, Denis and Li, William H. and Romasky, Brianna and Stavitski, Eli and Walter, Andrew L. and Moeller, Paul and Nash, Boaz and Islegen-Wojdyla, Antoine},
     journal  = {Journal of Synchrotron Radiation},
     title    = {{A general Bayesian algorithm for the autonomous alignment of beamlines}},
     year     = {2024},
     month    = {Nov},
     number   = {6},
     pages    = {1446--1456},
     volume   = {31},
     doi      = {10.1107/S1600577524008993},
     keywords = {Bayesian optimization, automated alignment, synchrotron radiation, digital twins, machine learning},
     url      = {https://doi.org/10.1107/S1600577524008993},
   }