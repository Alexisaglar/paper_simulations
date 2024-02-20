from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters_pv import parameters, LLE_parameters

# Constants
C_bat = 10000  # Battery capacity in kWh, for example
n_c = 0.95  # Charging efficiency
n_d = 0.95  # Discharging efficiency
P_max_c = 5000  # Charging power max
P_max_d = 5000  # Discharging power max
delta_t = 1  # Time step in hours
SoC_max = 100  # 100%
SoC_min = 0  # 0%

irradiance = pd.read_csv('data/irradiance_seasons.csv')
temperature = pd.read_csv('data/temperature_seasons.csv')
P_load = pd.read_csv('data/load_seasons.csv') * 1000

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

# # Sample data - replace with actual CSV data
# P_load = pd.read_csv('data/load_seasons.csv')
P_PV = PV_data['P_Si']

seasonal_loads = [P_load['winter'], P_load['spring'], P_load['summer'], P_load['autumn']]
P_load = pd.concat(seasonal_loads).reset_index(drop=True)

# Initialize variables
SoC = pd.Series(np.zeros(len(P_load)))  # State of Charge
P_bat = pd.Series(np.zeros(len(P_load)))  # Battery power

for t in range(1, len(P_load)):
    P_available = ((SoC_min - SoC.iloc[t-1]/100)) * C_bat * n_d * 1 # Power available
    P_required = (((SoC_max - SoC.iloc[t-1])/100) * C_bat) / n_c # Power required
    
    if P_load.iloc[t] > P_PV.iloc[t]:
        P_bat[t] = max(P_PV.iloc[t] - P_load.iloc[t], P_available, -P_max_d)
    elif P_load.iloc[t] < P_PV.iloc[t]:
        P_bat[t] = min(P_PV.iloc[t] - P_load.iloc[t], P_required, P_max_c)
    else:
        P_bat[t] = 0
    
    Z_bat = 1 if P_bat.iloc[t] > 0 else 0
    if Z_bat == 1 :
        SoC[t] = (SoC.iloc[t-1] + (n_c * P_bat.iloc[t] * delta_t)) / C_bat * 100
    else:
        SoC[t] = SoC.iloc[t-1] + (P_bat.iloc[t] * delta_t/n_d )/ C_bat * 100
# Calculate P_h^t
P_h = P_load - P_PV - P_bat  # Assuming first column has the data

# Output results
results = pd.DataFrame({
    'P_load': P_load,
    'P_PV': P_PV,
    'P_bat': P_bat,
    'SoC': SoC,
    'P_h': P_h
})


print(results)

# Set the figure size and resolution
plt.figure(figsize=(8, 4), dpi=300)
# Set the plot font to Arial, which is a sans-serif font
plt.rc('font', family='Arial')
# Plotting the seasonal adjustments
results['P_load'].plot(label='P_load', lw=2)
results['P_h'].plot(label='P_h', lw=2)
# results['SoC'].plot(label='SoC', lw=2)
results['P_PV'].plot(label='P_PV', lw=2)
# Labeling the axes with a larger font for clarity
plt.xlabel('Time (H)', fontsize=12)
plt.ylabel('Total Power Consumption (kW)', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
# Save the plot as a high-resolution PNG file
# plt.savefig('IEEE_formatted_plot.png', format='png')
plt.show()

# results['P_load'].plot(label='P_load', lw=2)
results['SoC'].plot(label='SoC', lw=2)
plt.show()
results['P_bat'].plot(label='SoC', lw=2)
plt.show()