##############################################################################
# Parameters
##############################################################################

graphene_params = {  # energies in milli electron volts
    "gamma0": 3160,
    "gamma1": 381,
    "gamma2": -15,
    "gamma3": 380,
    "gamma4": 140,
    "U": 50.0,  # interlayer bias
    "delta": 0.0, # 22.0
    "Delta": 0.0,
}

# Another parameter set for trilayer case
graphene_params_TLG = {  # energies in milli electron volts
    "gamma0": 3160,         # exp paper: 3100
    "gamma1": 381, # 390 ?  # exp paper: 380   # Zhang&MacDonald: 502
    "gamma2": -15, # -20 ?  # exp paper: -15   # Zhang&MacDonald: -17.1
    "gamma3": 380, # 315 ?  # exp paper: -290  # Zhang&MacDonald: -377
    "gamma4": 140, # 44 ?   # exp paper: -141  # Zhang&MacDonald: -99
    "U": 30.0,  # interlayer bias
    "delta": -10.5,         # exp paper: -10.5  # Zhang&MacDonald: -1.4
    "Delta": -2.3,          # exp paper: -2.3   # Zhang&MacDonald: 0 ?
}

graphene_params_TLG = {  # energies in milli electron volts
    "gamma0": 3160,         # exp paper: 3100
    "gamma1": 380,  # exp paper: 380   # Zhang&MacDonald: 502
    "gamma2": -15,  # exp paper: -15   # Zhang&MacDonald: -17.1
    "gamma3": -290,  # exp paper: -290  # Zhang&MacDonald: -377
    "gamma4": 141,  # exp paper: -141  # Zhang&MacDonald: -99
    "U": 30.0,  # interlayer bias
    "delta": -10.5/2,         # exp paper: -10.5  # Zhang&MacDonald: -1.4
    "Delta": -2.3/2,          # exp paper: -2.3   # Zhang&MacDonald: 0 ?
}

graphene_params_BLG = {  # energies in milli electron volts
    "gamma0": 3160,         # exp paper: 3100
    "gamma1": 380,  # exp paper: 380   # Zhang&MacDonald: 502
    "gamma2": -15,  # exp paper: -15   # Zhang&MacDonald: -17.1
    "gamma3":  -283,  # exp paper: -290  # Zhang&MacDonald: -377
    "gamma4": 138,  # exp paper: -141  # Zhang&MacDonald: -99
    "U": 0.0,  # interlayer bias
    "Delta": 0,          # exp paper: -2.3   # Zhang&MacDonald: 0 ?
    "delta": 15
}

graphene_params_4LG = {  # energies in milli electron volts
    "gamma0": 3160,         # exp paper: 3100
    "gamma1": 380,  # exp paper: 380   # Zhang&MacDonald: 502
    "gamma2": -15,  # exp paper: -15   # Zhang&MacDonald: -17.1
    "gamma3": -290,  # exp paper: -290  # Zhang&MacDonald: -377
    "gamma4": 141,  # exp paper: -141  # Zhang&MacDonald: -99
    "gamma5": 40, # CL: This is new to 4L graphene
    "U": 0.0,  # interlayer bias
    "Delta": 0.0,         # exp paper: -2.3   # Zhang&MacDonald: 0 ?
    "delta": 40.8,         # ENERGY DIFFERENCE BETWEEN DIMER AND NON-DIMER SITES
}