from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

irradiance = pd.read_csv('data/irradiance_seasons.csv')
temperature = pd.read_csv('data/temperature_seasons.csv')
P_load = pd.read_csv('data/load_seasons.csv')

def PV_power_generation(irradiance, temperature, parameters, LLE_parameters):
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
    
    # Calculate P_SI
    PV_data = pd.DataFrame()
    PV_data['P_Si'] = P_mp * parameters['series_cell'] * parameters['parallel_cell']

    # Adding irradiance and temperature as columns
    PV_data['irradiance'], PV_data['temperature'] = irradiance, temperature

    # Calculating delta, phi, beta values for P_LLE
    PV_data['phi'] = ((-1/100) * irradiance) + LLE_parameters['PCE_ref']
    PV_data['beta'] = 1
    PV_data['delta'] = (PV_data['phi'] * PV_data['beta'])/ LLE_parameters['PCE_min']
    PV_data['P_LLE'] = PV_data['P_Si'] * PV_data['delta']

    return PV_data

# Calculate power output for Si and LLE
PV_data = PV_power_generation(irradiance['GHI'], temperature['t2m'], parameters, LLE_parameters)

# Initial variables
p_battery, p_house = 0, 0
P_SI = PV_data['P_Si']
P_LLE = PV_data['P_lle']
C_discharge, C_charge = 5, 5
n_discharge, n_charge = 0.95, 0.95
SoC_max, SoC_min = 100, 0 
SoC = 0
C_bat = 10

for _, i in enumerate():
    # Calculate SoC
    if i > 0:
        if P_bat.iloc[i] > 0:
            SoC.iloc[i] = (SoC.iloc[i-1] + n_charge * P_bat.iloc[i] * 60) / C_bat
        else:
            SoC.iloc[i] = (SoC.iloc[i-1] +  (P_bat.iloc[i]/n_discharge) * 60 / C_bat)
    else:
        SoC.iloc[i] = 0

    # Calculate S1 and S2
    S1 = (SoC.iloc[i] - SoC_max) * C_bat * n_discharge
    S2 = (SoC.iloc[i] - SoC_max) * C_bat * n_discharge

    # power_house = power_load - power_pv - power_bat
    if P_load.iloc[i] > P_SI:
        P_bat = max(P_SI.iloc[i] - P_load.iloc[i], S1, C_discharge)
    else:
        P_bat = min(P_SI.iloc[i] - P_load.iloc[i], S2, C_charge)

    P_house_SI = P_load.iloc[i] - PV_data['P_si'] - P_bat_SI
    P_house_LLE = P_load.iloc[i] - PV_data['P_lle'] - P_bat_LLE
