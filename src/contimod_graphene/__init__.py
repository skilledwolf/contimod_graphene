"""
contimod_graphene: Standalone multilayer graphene Hamiltonians and utilities.

This package provides reusable low-level graphene-model tools, including
Bernal (ABA) and Rhombohedral (ABC) Hamiltonians, parameter sets, basis
metadata, and symmetry helpers.
"""

from .params import graphene_params, graphene_params_TLG, graphene_params_BLG, graphene_params_4LG
from . import params
from . import rhombohedral
from . import bernal
from . import basis
from . import symmetry

__all__ = [
    "params",
    "rhombohedral",
    "bernal",
    "basis",
    "symmetry",
    "graphene_params",
    "graphene_params_TLG",
    "graphene_params_BLG",
    "graphene_params_4LG",
]
