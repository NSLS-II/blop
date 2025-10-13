
.. warning::

    **Important Notice for upcoming v1.0.0 release**

    Major changes are expected in the v1.0.0 release of Blop. We are removing the older way of using agents and performing optimization in favor of using the Ax agent backend. The legacy agent interface will be deprecated, and users are encouraged to migrate to the new `blop.ax.Agent` interface for all optimization workflows. Please refer to the :doc:`tutorials`, :doc:`how-to-guides`, and :doc:`explanation` sections for the new interface.
    
    Ax will be the primary backend for optimization and experiment tracking. We will also be adding support for the `gest-api <https://github.com/campa-consortium/gest-api>`_ standard with integration into Ax experiment tracking.


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
