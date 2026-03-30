"""
contimod_graphene: Standalone multilayer graphene Hamiltonians and utilities.

This package provides reusable low-level graphene-model tools, including
Bernal (ABA) and Rhombohedral (ABC) Hamiltonians, validated parameter sets,
immutable model objects, basis metadata, and symmetry helpers.
"""

from .models import ABAMultilayer, ABCMultilayer, BernalMultilayer, RhombohedralMultilayer
from .params import (
    GrapheneTBParameters,
    graphene_params,
    graphene_params_TLG,
    graphene_params_BLG,
    graphene_params_4LG,
    list_parameter_sets,
    load_parameter_set,
)
from . import params
from . import rhombohedral
from . import bernal
from . import basis
from . import symmetry
from . import models

__all__ = [
    "ABAMultilayer",
    "ABCMultilayer",
    "BernalMultilayer",
    "GrapheneTBParameters",
    "RhombohedralMultilayer",
    "params",
    "models",
    "rhombohedral",
    "bernal",
    "basis",
    "symmetry",
    "graphene_params",
    "graphene_params_TLG",
    "graphene_params_BLG",
    "graphene_params_4LG",
    "list_parameter_sets",
    "load_parameter_set",
]
