"""
contimod_graphene: Standalone multilayer graphene Hamiltonians and utilities.

This package provides reusable low-level graphene-model tools, including
Bernal (ABA) and Rhombohedral (ABC) Hamiltonians, validated parameter sets,
immutable model objects, basis metadata, and symmetry helpers.
"""

from importlib.metadata import PackageNotFoundError, version as package_version

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

try:
    from ._version import version as __version__
except ImportError:
    try:
        __version__ = package_version("contimod_graphene")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = [
    "ABAMultilayer",
    "ABCMultilayer",
    "BernalMultilayer",
    "GrapheneTBParameters",
    "RhombohedralMultilayer",
    "__version__",
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
