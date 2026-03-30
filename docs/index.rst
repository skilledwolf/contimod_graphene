Welcome to contimod_graphene
============================

**contimod_graphene** is a standalone Python package for multilayer graphene Hamiltonians, parameter sets, basis metadata, and related single-particle utilities.

The recommended entry point is the top-level API:

- ``GrapheneTBParameters``
- ``BernalMultilayer``
- ``RhombohedralMultilayer``

Low-level kernel modules remain available for advanced use, but the user guide below leads with the standalone model/parameter surface rather than downstream ``contimod`` integration.

Features
--------

*   **Bernal (ABA) Stacking**: Support for N-layer Bernal stacked systems.
*   **Rhombohedral (ABC) Stacking**: Support for N-layer Rhombohedral stacked systems.
*   **Landau Levels**: Construction of Hamiltonians in a magnetic field using the Landau Level basis.
*   **Effective Models**: 2-band effective Hamiltonians for ABC systems.

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
