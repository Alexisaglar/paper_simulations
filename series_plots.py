from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parameters_pv import parameters, LLE_parameters

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

winter_start, spring_start, summer_start, autumn_start = 0, 1440, 2880, 4320
winter_end, spring_end, summer_end, autumn_end = 1439, 2879, 4319, 5759

# Combining seasonal loads
seasonal_loads = [P_load['winter'], P_load['spring'], P_load['summer'], P_load['autumn']]
P_load = pd.concat(seasonal_loads).reset_index(drop=True)

# Initialize variables
SoC = pd.Series(np.zeros(len(P_load)))  # State of Charge
P_bat = pd.Series(np.zeros(len(P_load)))  # Battery power

# color scheme
colors = {
    'P_PV': '#FD841F',    # Orange for solar PV generation
    'P_G2H': '#40679E',   # Green for generation from Grid to Hydrogen
    'P_H2G': '#4CACBC',   # Cyan for Hydrogen to Grid
    'Bat_charging': '#AFC8AD' ,   # Yellow for battery storage
    'Bat_discharging': '#527853' ,   # Yellow for battery storage
}

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

def calculate_self_consumption(P_PV):
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
    P_h = P_load - P_PV + P_bat  
    # Divide P_h in P_H2G and P_G2H
    P_H2G = pd.Series(np.zeros(len(P_h)))  # Battery power
    P_G2H = pd.Series(np.zeros(len(P_h)))  # Battery power
    for i in range(len(P_h)):
        if P_h[i] > 0:
            P_G2H[i] = P_h[i] 
        else:
            P_H2G[i] = P_h[i] 
    # Divide in bat_charge and discharge for better graphing
    Bat_charge = pd.Series(np.zeros(len(P_bat)))  # Battery power
    Bat_discharge = pd.Series(np.zeros(len(P_bat)))  # Battery power
    for i in range(len(P_bat)):
        if P_bat[i] > 0:
            Bat_charge[i] = P_bat[i] 
        else:
            Bat_discharge[i] = P_bat[i] 
    return SoC, P_bat, P_available, P_required, P_h, P_G2H, P_H2G, Bat_charge, Bat_discharge

def stack_plot(P_load, P_G2H, P_H2G, Bat_charge, Bat_discharge, P_PV, name):        
    # Plot figure
    plt.figure(figsize=(12, 6))
    # Increase the overall font sizes for readability
    plt.rcParams.update({'font.size': 20})  # Adjust as needed for IEEE guidelines
    plt.stackplot(P_load.index, 
                (P_load - P_G2H)/1000,
                P_G2H/1000, 
                abs(P_H2G)/1000, 
                Bat_charge/1000, 
                Bat_discharge/1000, 
                labels=['PV to Load','Grid to Home', 'Home to Grid', 'Battery Charging', 'Battery Discharging'], 
                colors=colors.values(),
                alpha=0.8)

    plt.plot(P_load.index, 
            P_load/1000, label='Load Power', 
            color='black')
    plt.plot(P_PV.index, 
            P_PV/1000, 
            label='PV Power', color='#FD841F')

    ax = plt.gca()  # Get the current Axes instance
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    season_centers = [750, 750+1440, (1440*2)+750, (1440*3)+750,]  # Assuming each section is 6 'ticks' wide

    # Sample season names
    season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
    # Add text annotations for each season
    for x, season in zip(season_centers, season_names):
        plt.text(x, 4.15, season, horizontalalignment='center', fontsize=20)  # Adjust `max_power + 200` as needed

    season_starts = [0, 1440, 2880, 4320, 5760]
    season_starts = [1440, 2880, 4320]
    season_midpoints = [720, 2160, 3600, 5040]
    season_labels = ['Winter', 'Spring', 'Summer', 'Autumn']
    # Set the limits for a cleaner look (optional)
    tick_locations = np.arange(0, 5759 + 1, 360)
    # Generate tick labels, resetting to "0h" after every 24 hours (1440 minutes)
    tick_labels = [f"{(minutes // 60) % 24}h" for minutes in tick_locations]

    # Update the plt.xticks call to use hour_ticks
    plt.xticks(tick_locations, tick_labels)
    plt.xlim(0, 5760)
    plt.ylim([0, 4])  # for example, to set the y-limit to 20% above max PV power
    for tick in season_starts:
        plt.axvline(x=tick, color='k', linestyle='--')  # k is color black, you can choose any color

    # Set the thickness of the plot border lines
    for _, spine in ax.spines.items():
        spine.set_linewidth(1.5)

    # Increase the line width of the tick marks for visibility
    ax.tick_params(which='both', width=2)  # Applies to both major and minor ticks
    ax.tick_params(which='major', length=7)  # Only major ticks
    ax.tick_params(which='minor', length=4, color='gray')  # Only minor ticks

    # Add labels and title with a larger font size
    plt.xlabel('Time (h)', fontsize=20)
    plt.ylabel('Power (kW)', fontsize=20)
    # Make sure the legend is readable
    plt.legend(fontsize='20', frameon=True, handlelength=0.5, labelspacing=0.2, handletextpad=0.3, borderpad=0.2, loc='upper left')  # Adjust the location and size as needed
    plt.show()

def calculate_total_energy(P_load, P_G2H, P_H2G, Bat_charge, Bat_discharge, P_PV):
    E_Load = (P_load/60)/1000
    E_PV = (P_PV/60)/1000
    E_G2H = (P_G2H/60)/1000
    E_H2G = (P_H2G/60)/1000
    E_charge = (Bat_charge/60)/1000
    E_discharge = (Bat_discharge/60)/1000
    return E_Load, E_PV, E_G2H, E_H2G, E_charge, E_discharge

def interpolate_list(array1, array2):
    scale_points = [1.2, 1.4, 1.6, 1.8]
    deltas = array2 - array1
    interpolated_results = []
    scale_1 = 1.0
    scale_2 = 2.0
    for scale_point in scale_points:
        scale_factor = (scale_point - scale_1)/ (scale_2 -scale_1)
        interpolate_values = array1 + deltas * scale_factor
        interpolated_results.append(interpolate_values)
    return np.array(interpolated_results)

def data_interpolation(E_PV1, E_PV2, E_G2H1, E_G2H2, E_H2G1, E_H2G2, E_charge1, E_charge2, E_discharge1, E_discharge2):
    E_PV = interpolate_list(E_PV1, E_PV2)
    E_G2H = interpolate_list(E_G2H1, E_G2H2)
    E_H2G = interpolate_list(E_H2G1, E_H2G2)
    E_charge = interpolate_list(E_charge1, E_charge2)
    E_discharge = interpolate_list(E_discharge1, E_discharge2)

    return E_PV, E_G2H, E_H2G, E_charge, E_discharge

def data_per_season(E_PV, E_G2H, E_H2G, E_charge, E_discharge, season_start, season_end):
    E_PV_season = [E_PV_Si[season_start:season_end].sum(), E_PV[0][season_start:season_end].sum(),
                    E_PV[1][season_start:season_end].sum(), E_PV[2][season_start:season_end].sum(),
                    E_PV[3][season_start:season_end].sum(),  E_PV_LLE[season_start:season_end].sum()]
    E_G2H_season = [E_G2H_Si[season_start:season_end].sum(), E_G2H[0][season_start:season_end].sum(),
                    E_G2H[1][season_start:season_end].sum(), E_G2H[2][season_start:season_end].sum(),
                    E_G2H[3][season_start:season_end].sum(),  E_G2H_LLE[season_start:season_end].sum()]
    E_H2G_season = [E_H2G_Si[season_start:season_end].sum(), E_H2G[0][season_start:season_end].sum(),
                    E_H2G[1][season_start:season_end].sum(), E_H2G[2][season_start:season_end].sum(), 
                    E_H2G[3][season_start:season_end].sum(), E_H2G_LLE[season_start:season_end].sum()]
    E_discharge_season = [E_discharge_Si[season_start:season_end].sum(), E_discharge[0][season_start:season_end].sum(),
                        E_discharge[1][season_start:season_end].sum(), E_discharge[2][season_start:season_end].sum(),
                        E_discharge[3][season_start:season_end].sum(), E_discharge_LLE[season_start:season_end].sum()]
    E_charge_season = [E_charge_Si[season_start:season_end].sum() + E_load, E_charge[0][season_start:season_end].sum() + E_load, 
                       E_charge[1][season_start:season_end].sum() + E_load, E_charge[2][season_start:season_end].sum() + E_load, 
                       E_charge[3][season_start:season_end].sum() + E_load, E_charge_LLE[season_start:season_end].sum() + E_load]

    return E_PV_season, E_G2H_season, E_H2G_season, E_charge_season, E_discharge_season

# Calculate power output for Si and LLE
PV_data = PV_power_generation(irradiance['GHI'], temperature['t2m'], parameters, LLE_parameters)

# Calculate self consumption for each technology
SoC_Si, P_bat_Si, P_available_Si, P_required_Si, P_h_Si, P_G2H_Si, P_H2G_Si, Bat_charge_Si, Bat_discharge_Si = calculate_self_consumption(PV_data['P_Si'])
SoC_LLE, P_bat_LLE, P_available_LLE, P_required_LLE, P_h_LLE, P_G2H_LLE, P_H2G_LLE, Bat_charge_LLE, Bat_discharge_LLE = calculate_self_consumption(PV_data['P_LLE'])

# Plot both technologies
stack_plot(P_load, P_G2H_Si, P_H2G_Si, Bat_charge_Si, Bat_discharge_Si, PV_data['P_Si'], 'Silicon')
stack_plot(P_load, P_G2H_LLE, P_H2G_LLE, Bat_charge_LLE, Bat_discharge_LLE, PV_data['P_LLE'], 'LLE')

# Calculate energy consumed
E_Load_Si, E_PV_Si, E_G2H_Si, E_H2G_Si, E_charge_Si, E_discharge_Si = calculate_total_energy(P_load, P_G2H_Si, P_H2G_Si, Bat_charge_Si, Bat_discharge_Si, PV_data['P_Si'])
E_Load_LLE, E_PV_LLE, E_G2H_LLE, E_H2G_LLE, E_charge_LLE, E_discharge_LLE = calculate_total_energy(P_load, P_G2H_LLE, P_H2G_LLE, Bat_charge_LLE, Bat_discharge_LLE, PV_data['P_LLE'])

E_load = (P_load[winter_start:winter_end].sum()/60)/1000 

# interpolate values between known values
E_PV, E_G2H, E_H2G, E_charge, E_discharge = data_interpolation(E_PV_Si, E_PV_LLE, E_G2H_Si, E_G2H_LLE, E_H2G_Si, E_H2G_LLE, E_charge_Si, E_charge_LLE, E_discharge_Si, E_discharge_LLE)
PV_3d=np.array(E_PV_Si), E_PV[0], E_PV[1], E_PV[2], E_PV[3], np.array(E_PV_LLE)
G2H_3d=[np.array(E_G2H_Si), E_G2H[0], E_G2H[1], E_G2H[2], E_G2H[3], np.array(E_G2H_LLE)]
discharge_3d=[abs(np.array(E_discharge_Si)), abs(E_discharge[0]), abs(E_PV[1]), abs(E_PV[2]), abs(E_PV[3]), abs(np.array(E_PV_LLE))]
# G2H_[np.array(E_PV_Si), E_PV[0], E_PV[1], E_PV[2], E_PV[3], np.array(E_PV_LLE)]
Total_E_PV, Total_E_G2H, Total_E_H2G, Total_E_charge, Total_E_discharge = data_per_season(E_PV, E_G2H, E_H2G, E_charge, E_discharge, winter_start, winter_end)


# Plot
delta_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # The scaling parameter values
plt.figure(figsize=(7, 10))
plt.stackplot(delta_values, Total_E_G2H/E_load, np.abs(Total_E_discharge)/E_load, Total_E_PV/E_load,
              labels=['E_G2H', 'E_discharge', 'E_PV'],
              colors=['#40679E', '#527853', '#FD841F'])
plt.plot([1,2], [1,1], label='Load', color='black')

plt.fill_between(delta_values, [1,1,1,1,1,1], Total_E_charge/E_load, color='#AFC8AD', hatch='O', alpha =0.5, label='E_charge' )
ax = plt.gca()  # Get the current Axes instance
plt.xlim(1, 2)
plt.ylim([0, 1.4])  # for example, to set the y-limit to 20% above max PV power
for _, spine in ax.spines.items():
    spine.set_linewidth(1.5)

# Increase the line width of the tick marks for visibility
ax.tick_params(which='both', width=2)  # Applies to both major and minor ticks
ax.tick_params(which='major', length=7)  # Only major ticks
ax.tick_params(which='minor', length=4, color='gray')  # Only minor ticks

# Add labels and title with a larger font size
plt.xlabel('δ_material', fontsize=20)
plt.ylabel('Energy % to Load', fontsize=20)
plt.legend(fontsize='20', frameon=True, handlelength=0.5, labelspacing=0.1, handletextpad=0.3, borderpad=0.1, loc='upper left')  # Adjust the location and size as needed
plt.show()
