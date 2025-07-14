=========
blop
=========

.. image:: https://github.com/NSLS-II/blop/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/NSLS-II/blop/actions/workflows/testing.yml


.. image:: https://img.shields.io/pypi/v/blop.svg
        :target: https://pypi.python.org/pypi/blop

.. image:: https://img.shields.io/conda/vn/conda-forge/blop.svg
        :target: https://anaconda.org/conda-forge/blop

Beamline Optimization Tools

* Free software: 3-clause BSD license
* Documentation: https://NSLS-II.github.io/blop.

Citation
--------

If you use this package in your work, please cite the following paper:

```bibtex
@Article{Morris2024,
  author   = {Morris, Thomas W. and Rakitin, Max and Du, Yonghua and Fedurin, Mikhail and Giles, Abigail C. and Leshchev, Denis and Li, William H. and Romasky, Brianna and Stavitski, Eli and Walter, Andrew L. and Moeller, Paul and Nash, Boaz and Islegen-Wojdyla, Antoine},
  journal  = {Journal of Synchrotron Radiation},
  title    = {{A general Bayesian algorithm for the autonomous alignment of beamlines}},
  year     = {2024},
  month    = {Nov},
  number   = {6},
  pages    = {1446--1456},
  volume   = {31},
  abstract = {Autonomous methods to align beamlines can decrease the amount of time spent on diagnostics, and also uncover better global optima leading to better beam quality. The alignment of these beamlines is a high-dimensional expensive-to-sample optimization problem involving the simultaneous treatment of many optical elements with correlated and nonlinear dynamics. Bayesian optimization is a strategy of efficient global optimization that has proved successful in similar regimes in a wide variety of beamline alignment applications, though it has typically been implemented for particular beamlines and optimization tasks. In this paper, we present a basic formulation of Bayesian inference and Gaussian process models as they relate to multi-objective Bayesian optimization, as well as the practical challenges presented by beamline alignment. We show that the same general implementation of Bayesian optimization with special consideration for beamline alignment can quickly learn the dynamics of particular beamlines in an online fashion through hyperparameter fitting with no prior information. We present the implementation of a concise software framework for beamline alignment and test it on four different optimization problems for experiments on X-ray beamlines at the National Synchrotron Light Source II and the Advanced Light Source, and an electron beam at the Accelerator Test Facility, along with benchmarking on a simulated digital twin. We discuss new applications of the framework, and the potential for a unified approach to beamline alignment at synchrotron facilities.},
  doi      = {10.1107/S1600577524008993},
  keywords = {Bayesian optimization, automated alignment, synchrotron radiation, digital twins, machine learning},
  url      = {https://doi.org/10.1107/S1600577524008993},
}
```
