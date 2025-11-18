"""
Standard parameters for graphene tight-binding models.

Energies are in milli-electron volts (meV).
"""

graphene_params = {
    "gamma0": 3160,
    "gamma1": 381,
    "gamma2": -15,
    "gamma3": 380,
    "gamma4": 140,
    "U": 50.0,
    "delta": 0.0,
    "Delta": 0.0,
}

graphene_params_TLG = {
    "gamma0": 3160,
    "gamma1": 380,
    "gamma2": -15,
    "gamma3": -290,
    "gamma4": 141,
    "U": 30.0,
    "delta": -10.5/2,
    "Delta": -2.3/2,
}

graphene_params_BLG = {
    "gamma0": 3160,
    "gamma1": 380,
    "gamma2": -15,
    "gamma3": -283,
    "gamma4": 138,
    "U": 0.0,
    "Delta": 0,
    "delta": 15
}

graphene_params_4LG = {
    "gamma0": 3160,
    "gamma1": 380,
    "gamma2": -15,
    "gamma3": -290,
    "gamma4": 141,
    "gamma5": 40,
    "U": 0.0,
    "Delta": 0.0,
    "delta": 40.8,
}
