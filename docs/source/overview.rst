Overview
========

.. warning::

    **Important Notice for v1.0.0**

    Major changes are expected in the v1.0.0 release of Blop. We are removing the older way of using agents and performing optimization in favor of using the Ax agent backend. The legacy agent interface will be deprecated, and users are encouraged to migrate to the new `blop.ax.Agent` interface for all optimization workflows. Please refer to the migration guide and updated documentation for the new interface.

What is Blop?
-------------

**Blop** (Beamline Optimization Package) is a Python library for autonomous optimization of synchrotron beamlines using machine learning and Bayesian optimization techniques. It enables researchers to automatically tune experimental parameters to optimize objectives like beam intensity, focus quality, or any other measurable beamline characteristics.

The package addresses a critical need in synchrotron science: the complex, time-consuming process of manually optimizing numerous interdependent beamline parameters. Traditional manual tuning is not only labor-intensive but often suboptimal, as the parameter space is typically high-dimensional and contains complex correlations that are difficult for humans to navigate efficiently.

Why is Blop Useful?
-------------------

Blop provides several key advantages for synchrotron experiments:

**Autonomous Optimization**
  Automatically explores parameter space using intelligent sampling strategies, reducing the need for manual intervention and expert knowledge about parameter relationships.

**Bayesian Intelligence** 
  Uses Gaussian Process models and sophisticated acquisition functions to make informed decisions about which parameters to try next, minimizing the number of experiments needed to find optimal settings.

**Multi-objective Optimization**
  Simultaneously optimizes multiple competing objectives (e.g., maximize intensity while minimizing beam size), helping find Pareto-optimal solutions.

**Integration with Existing Workflows**
  Seamlessly integrates with the Bluesky ecosystem, working with existing experimental setups, data collection systems, and analysis pipelines.

**Real-time Adaptation**
  Continuously learns from experimental results and adapts its optimization strategy, making it robust to changing experimental conditions.

**Reproducible Science**
  Provides detailed tracking of optimization history, parameters tested, and results obtained, ensuring reproducible and auditable optimization processes.

Project Structure
-----------------

Blop is built on two foundational technologies that provide complementary capabilities:

Built on Ax Platform
~~~~~~~~~~~~~~~~~~~~

At its core, Blop leverages `Ax <https://ax.dev/>`_ (Meta's Adaptive Experimentation Platform) for Bayesian optimization:

- **Modern Optimization Backend**: Ax provides state-of-the-art Bayesian optimization algorithms, including support for multi-objective optimization, constraints, and advanced acquisition functions.

- **Experiment Management**: Robust experiment tracking, parameter configuration, and results management through Ax's proven infrastructure.

- **Scalability**: Ax's architecture supports everything from simple single-objective problems to complex multi-objective, multi-fidelity, and contextual optimization scenarios.

- **Research-Grade Algorithms**: Access to cutting-edge optimization techniques including Expected Improvement, Upper Confidence Bound, and multi-objective algorithms like Expected Hypervolume Improvement.

Integration with Bluesky Ecosystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blop provides native integration with the Bluesky ecosystem for experimental orchestration:

- **Bluesky Plans**: Automatically generates and executes Bluesky measurement plans (like `list_scan`) based on optimization recommendations.

- **RunEngine Integration**: Works directly with Bluesky's RunEngine for experiment execution, ensuring compatibility with existing beamline control systems.

- **Databroker Compatibility**: Seamlessly reads experimental data from Databroker/Tiled instances for analysis and optimization feedback.

- **Device Support**: Works with standard Ophyd devices (motors, detectors) commonly used in Bluesky environments.

- **Metadata Tracking**: Maintains full integration with Bluesky's metadata and experiment tracking capabilities.

Core Components
~~~~~~~~~~~~~~~

The package is organized around several key concepts:

**DOFs (Degrees of Freedom)**
  Define the controllable parameters in your experiment (motors, voltages, etc.) along with their search domains and constraints.

**Objectives** 
  Specify what you want to optimize (maximize beam intensity, minimize beam size, etc.) along with targets and constraints.

**Agent**
  The central coordinator that combines DOFs, objectives, and experimental infrastructure to perform autonomous optimization.

**Digestion Functions**
  Custom analysis functions that process raw experimental data into objective values that the optimizer can use.

This architecture provides a clean separation between the optimization intelligence (Ax), experimental orchestration (Bluesky), and domain-specific analysis (digestion functions), making Blop both powerful and flexible for a wide range of beamline optimization scenarios.
