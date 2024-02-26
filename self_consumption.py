from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters_pv import parameters, LLE_parameters

colors = {
    'P_load': 'slategray',
    'P_G2H': 'forestgreen',  # Green for generation from Grid to Hydrogen
    'P_PV': 'gold',          # Yellowish for solar PV generation
    'P_H2G': 'royalblue',    # Blue for Hydrogen to Grid
    'P_bat': 'mediumpurple'  # Purple for battery storage
}

# Constants
C_bat = 5000  # Battery capacity in kWh, for example
n_c = 0.95  # Charging efficiency
n_d = 0.95  # Discharging efficiency
P_max_c = 2500  # Charging power max
P_max_d = 2500  # Discharging power max
delta_t = 60  # Time step in hours
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
    P_available = ((SoC_min - SoC.iloc[t-1]/100)) * (C_bat * delta_t) * n_d# Power available
    P_required = ((((SoC_max - SoC.iloc[t-1])/100) * (C_bat * delta_t)) / (n_c))  # Power required
    
    if t == 1440 or t == 2880 or t == 4320 or t == 0: 
        SoC[t-1] = 0 
    
    if P_load.iloc[t] > P_PV.iloc[t]:
        P_bat[t] = max(P_PV.iloc[t] - P_load.iloc[t], P_available, -P_max_d)
    # elif P_load.iloc[t] < P_PV.iloc[t]:
    else:
        P_bat[t] = min(P_PV.iloc[t] - P_load.iloc[t], P_required, P_max_c)
        # P_bat[t] = 0
    
    Z_bat = 1 if P_bat.iloc[t] > 0 else 0
    if Z_bat == 1 :
        SoC[t] = (SoC.iloc[t-1] + (n_c * ((P_bat.iloc[t] / delta_t) / C_bat) * 100))
    else:
        SoC[t] = SoC.iloc[t-1] + (((P_bat.iloc[t] / delta_t * n_d ) / C_bat )* 100) 
        
# Calculate P_h^t , P_H2G, P_G2H
P_h = P_load - P_PV - P_bat  



# Initialize variables
P_H2G = pd.Series(np.zeros(len(P_h)))  # Battery power
P_G2H = pd.Series(np.zeros(len(P_h)))  # Battery power

for i in range(len(P_h)):
    if P_h[i] > 0:
        P_G2H[i] = P_h[i] + P_bat[i]
    else:
        P_H2G[i] = P_h[i] + P_bat[i]

# Output results
results = pd.DataFrame({
    'P_load': P_load,
    'P_PV': P_PV,
    'P_bat': P_bat,
    'SoC': SoC,
    'P_h': P_h,
    'P_H2G': P_H2G,
    'P_G2H': P_G2H,
})

print(results)

# Improved color scheme
colors = {
    'P_load': '#4e79a7',  # Blue for load
    'P_G2H': '#59a14f',   # Green for generation from Grid to Hydrogen
    'P_PV': '#f28e2b',    # Orange for solar PV generation
    'P_H2G': '#76b7b2',   # Cyan for Hydrogen to Grid
    'P_bat': '#edc948'    # Yellow for battery storage
}

# Improved plot aesthetics
plt.figure(figsize=(10, 6))
# plt.stackplot(results.index, results['P_G2H'], abs(results['P_bat']), (results['P_load']-results['P_G2H']), labels=['P_G2H', 'P_H2G', 'P_PV'], colors=colors.values(), alpha=0.8)
plt.stackplot(results.index, results['P_G2H'], results['P_PV'] + results['P_H2G'], abs(results['P_H2G']), labels=['P_G2H', 'P_PV', 'P_H2G'], colors=colors.values(), alpha=0.8)
plt.plot(results.index, results['P_load'], label='P_load', color='black')
plt.plot(results.index, results['P_PV'], label='P_PV', color='red')
plt.title('Power Distribution Over Time', fontsize=16)
plt.xlabel('Time (h)', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

# Save the figure with high quality
plt.savefig('power_distribution.png', dpi=300)

plt.show()

print(results)