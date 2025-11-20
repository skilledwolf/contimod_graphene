"""
contimod_graphene: Helper package for contimod.

This package provides Hamiltonian construction tools for multilayer graphene systems,
including Bernal (ABA) and Rhombohedral (ABC) stacking.
"""

from .params import graphene_params, graphene_params_TLG, graphene_params_BLG, graphene_params_4LG
from . import params
from . import rhombohedral
from . import bernal
from . import basis

__all__ = [
    "params",
    "rhombohedral",
    "bernal",
    "basis",
    "graphene_params",
    "graphene_params_TLG",
    "graphene_params_BLG",
    "graphene_params_4LG",
]
