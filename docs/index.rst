Welcome to contimod_graphene
============================

**contimod_graphene** is a standalone Python package for multilayer graphene Hamiltonians, parameter sets, basis metadata, and related single-particle utilities.

If you only read one page, start with the user guide: it gives the shortest working recipes, the core equations, and concrete output examples.

The recommended API entry points are:

- ``GrapheneTBParameters``
- ``BernalMultilayer``
- ``RhombohedralMultilayer``

Low-level kernel modules remain available for advanced use, but the docs below lead with the standalone model/parameter surface rather than downstream ``contimod`` integration.

Start Here
----------

- ``usage``: quickest API path, equations, conventions, and representative outputs
- ``examples``: figure-producing scripts and expected artifacts
- ``installation``: install and local-development notes

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
