===============
Release History
===============

v0.7.1 (2024-09-26)
-------------------
- Add simulated hardware.
- Added a method to prune bad data.

v0.7.0 (2024-05-13)
-------------------
- Added functionality for Pareto optimization.
- Support for discrete degrees of freedom.

v0.6.0 (2024-02-01)
-------------------
- More sophisticated targeting capabilities for different objectives.
- More user-friendly agent controls.

v0.5.0 (2023-11-09)
-------------------
- Added hypervolume acquisition and constraints.
- Better specification of latent dimensions.
- Implemented Monte Carlo acquisition functions.
- Added classes for DOFs and objectives.

v0.4.0 (2023-08-11)
-------------------

- Easier-to-use syntax when building the agent.
- Modular and stateful agent design for better usability.
- Added the ability to save/load both data and hyperparameters.
- Added passive degrees of freedom.
- Added a number of `test functions / artificial landscapes for optimization
  <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_.
- Updated the Sphinx documentation theme to `furo <https://github.com/pradyunsg/furo>`_.


v0.3.0 (2023-06-17)
-------------------

- Implemented multi-task optimization.
- Simplified the syntax on initializing the agent.
- Resolved issues discovered at NSLS-II ISS.


v0.2.0 (2023-04-25)
-------------------

- Rebased the Bayesian optimization models to be compatible with ``botorch`` code.
- Optimization objectives can be customized with ``experiment`` modules.
- Added optimization test functions for quicker testing and development.


v0.1.0 (2023-03-10)
-------------------

- Changed from using ``SafeConfigParser`` to ``ConfigParser``.
- Implemented the initial version of the GP optimizer.
- Updated the repo structure based on the new cookiecutter.
- Added tests to the CI.


v0.0.2 (2021-05-14)
-------------------

Fixed ``_run_flyers()`` for sirepo optimization.


v0.0.1 - Initial Release (2020-09-01)
-------------------------------------

Initial release of the Beamline Optimization library.

Used in:

- https://github.com/NSLS-II-TES/profile_simulated_hardware
- https://github.com/NSLS-II-TES/profile_sirepo

Planned:

- https://github.com/NSLS-II-TES/profile_collection
