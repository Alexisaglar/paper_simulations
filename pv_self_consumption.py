from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib as plt
from parameters_pv import parameters, LLE_parameters

# 1. Make Si-PV generation
#       P_PV = V_mpp * Impp
# 2. Calculate LLE-PV Based on Si-PV generation
#       P_lowPV = delta * alpha / P_PV
#       delta = f(irradiance)
#       alpha = f(temperature)
# 3. Calculate P_h^t which is based on PV generation, P_load and P_bat
#       P_h^t = P_load^t - P_PV^t - P_bat^t
# 4. Calculate SoC^t 
#       SoC = E_bat/E_cap)

def PV_power_generation(irradiance, temperature, PCE_min, parameters, LLE_parameters):
    # Preallocate the output array
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance=irradiance,
        temp_cell=temperature,
        alpha_sc=parameters['alpha_sc'],
        a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'],
        I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'],
        R_s=parameters['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )
    
    curve_info = pvsystem.singlediode(
        photocurrent=IL,
        saturation_current=I0,
        resistance_series=Rs,
        resistance_shunt=Rsh,
        nNsVth=nNsVth,
        ivcurve_pnts=100,
        method='lambertw'
    )
    
    # Calculate the module power
    P_mp = curve_info['i_mp'] * curve_info['v_mp']
    
    # Scale by the number of series and parallel panels
    P_out = P_mp * parameters['series_cell'] * parameters['parallel_cell']

    return P_out


