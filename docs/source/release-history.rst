===============
Release History
===============

v0.9.0 (2025-12-08)
-------------------

What's Changed
..............
* Protocols for optimization with Bluesky by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/185
* Evaluation functions by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/186
* Standard plans for optimization that work with protocols by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/187
* Refactor Ax agent to use new protocols by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/188
* Remove default evaluation options by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/189
* Direct AxOptimizer protocol impelementation by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/191
* Remove unused Ax helpers by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/193
* Simplified DOF classes for Ax backend by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/195
* Simpler Objective class, ScalarizedObjective, and OutcomeConstraints by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/196
* Protocol explanation by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/192
* New DOFConstraint class and Agent update by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/198
* Agent composition over inheritance by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/199
* Add how-to-guide for outcome constraints by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/200
* Extensive updates to reference docs by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/201
* Change furo theme -> pydata-sphinx-theme by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/202
* Update docstrings with much more detail by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/203


`Full Changelog <https://github.com/NSLS-II/blop/compare/v0.8.1...v0.9.0>`__

v0.8.1 (2025-11-06)
-------------------

What's Changed
..............
* Pin networkx and tabulate (required by python-tsp) by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/171
* Update blop conda-forge badge by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/172
* Baseline measurements by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/173
* Implement suggest & ingest from gest-api by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/174
* Move tests to tests/integration by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/176
* Improve Bluesky plans by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/177
* Include ellipses in doctest by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/179
* Parameter constraints by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/181


`Full Changelog <https://github.com/NSLS-II/blop/compare/v0.8.0...v0.8.1>`__

v0.8.0 (2025-10-13)
-------------------

What's Changed
..............
* DOC: add citation information to `README` by `@mrakitin <https://github.com/mrakitin>`_ in https://github.com/NSLS-II/blop/pull/119
* Ax Adapters for Blop and Minimal Agent Interface by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/126
* Adding Pixi by `@jessica-moylan <https://github.com/jessica-moylan>`_ in https://github.com/NSLS-II/blop/pull/135
* Ax custom generation strategies by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/136
* Ax Analyses by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/138
* Adding tiled comptability using dictionaries by `@jessica-moylan <https://github.com/jessica-moylan>`_ in https://github.com/NSLS-II/blop/pull/143
* Implement multitask models by `@thomaswmorris <https://github.com/thomaswmorris>`_ in https://github.com/NSLS-II/blop/pull/124
* Refactored tests to reduce redundancy (179 -> 35 tests) by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/148
* Update tutorial notebooks by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/147
* Deprecating older APIs by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/146
* Auto-generated API docs for Agent, AxAgent, DOFs, and Objectives by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/149
* Executable tutorials using jupyterlab by `@jessica-moylan <https://github.com/jessica-moylan>`_ in https://github.com/NSLS-II/blop/pull/154
* Update and simplify packaging by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/155
* Documentation Update by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/160
* Add docs on attaching data to experiments by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/162
* How-to guide for custom generation strategies by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/163

New Contributors
................
* `@jessica-moylan <https://github.com/jessica-moylan>`_ made their first contribution in https://github.com/NSLS-II/blop/pull/117

`Full Changelog <https://github.com/NSLS-II/blop/compare/v0.7.5...v0.8.0>`__

v0.7.5 (2025-06-18)
-------------------

What's Changed
..............
* Remove 'created' from release types by `@jennmald <https://github.com/jennmald>`_ in https://github.com/NSLS-II/blop/pull/105
* Refactor DOFs to fix trust domain behavior by `@thomaswmorris <https://github.com/thomaswmorris>`_ in https://github.com/NSLS-II/blop/pull/97
* Update installation.rst to reflect the version of python tested against by `@whs92 <https://github.com/whs92>`_ in https://github.com/NSLS-II/blop/pull/107
* Fix CI failures due to domain transforms by `@thomaswmorris <https://github.com/thomaswmorris>`_ in https://github.com/NSLS-II/blop/pull/108
* Update documentation for `Agent`, `DOF`, and `Objective` by `@thomaswmorris <https://github.com/thomaswmorris>`_ in https://github.com/NSLS-II/blop/pull/113
* Remove ortools as a dependency by `@thomaswmorris <https://github.com/thomaswmorris>`_ in https://github.com/NSLS-II/blop/pull/115
* Ax integrations with the Bluesky ecosystem by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/112

New Contributors
................
* `@whs92 <https://github.com/whs92>`_ made their first contribution in https://github.com/NSLS-II/blop/pull/107

`Full Changelog <https://github.com/NSLS-II/blop/compare/v0.7.4...v0.7.5>`__

v0.7.4 (2025-03-04)
-------------------
* Add missing files for documentation
* Fix trigger condition for releases on PyPI and documentation

v0.7.3 (2025-03-04)
-------------------
What's Changed
..............
* Fix documentation CI error by `@jennmald <https://github.com/jennmald>`_ in https://github.com/NSLS-II/blop/pull/84
* Fix fitness and constraint plots by `@jennmald <https://github.com/jennmald>`_ in https://github.com/NSLS-II/blop/pull/80
* Refactor: Make agent default compatible with Bluesky Adaptive by `@maffettone <https://github.com/maffettone>`_ in https://github.com/NSLS-II/blop/pull/86
* Ruff linter support; Removal of black, flake8, and isort by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/95
* Add type hints by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/87
* Remove Python 3.9 support by `@thopkins32 <https://github.com/thopkins32>`_ in https://github.com/NSLS-II/blop/pull/101
* Add XRT demo to blop tutorials by `@jennmald <https://github.com/jennmald>`_ in https://github.com/NSLS-II/blop/pull/102

New Contributors
................
* `@jennmald <https://github.com/jennmald>`_ made their first contribution in https://github.com/NSLS-II/blop/pull/84
* `@maffettone <https://github.com/maffettone>`_ made their first contribution in https://github.com/NSLS-II/blop/pull/86
* `@thopkins32 <https://github.com/thopkins32>`_ made their first contribution in https://github.com/NSLS-II/blop/pull/95

`Full Changelog <https://github.com/NSLS-II/blop/compare/v0.7.2...v0.7.3>`__

v0.7.2 (2025-01-31)
-------------------
- Renamed package in PyPI to `blop <https://pypi.org/project/blop/>`_.
- `bloptools <https://pypi.org/project/bloptools/>`_ is still avaliable on PyPI.

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
