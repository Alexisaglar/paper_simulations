from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parameters_pv import parameters, LLE_parameters

DELTA_VALUES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # The scaling parameter values

BETA_SI = 0.04
BETA_EPV = 0.025
TEMP_STC = 25
IRRADIANCE_STC = 1000
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

E_PV_season = pd.Series(len(DELTA_VALUES))
E_G2H_season = pd.Series(len(DELTA_VALUES))
E_H2G_season = pd.Series(len(DELTA_VALUES))
E_discharge_season = pd.Series(len(DELTA_VALUES))
E_charge_season = pd.Series(len(DELTA_VALUES))

SEASON_RANGES = np.array(
    (np.array([0, 1439]),
    np.array([1440, 2879]),
    np.array([2880, 4319]),
    np.array([4320, 5759]),
    ))

# color scheme
colors = {
    'P_PV': '#FD841F',    # Orange for solar PV generation
    'P_G2H': '#40679E',   # Green for generation from Grid to Hydrogen
    'P_H2G': '#4CACBC',   # Cyan for Hydrogen to Grid
    'Bat_charging': '#AFC8AD' ,   # Yellow for battery storage
    'Bat_discharging': '#527853' ,   # Yellow for battery storage
}
def calculate_si_power(irradiance, temperature):
    """
    Calculates the power output of a standard Si-PV system.
    Power is proportional to irradiance and decreases with temperature.
    """
    # Temperature degradation factor
    temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    
    # Power output calculation
    power = P_RATED * (irradiance / IRRADIANCE_STC) * temp_factor
    
    # Power cannot be negative
    return np.maximum(0, power)

def calculate_epv_power(irradiance, temperature):
    """
    Calculates the power output of the Low-Light Enhanced (LLE) PV system.
    This model includes gains from both low-light conditions and superior
    temperature performance, based on the thesis methodology.
    """
    # --- Irradiance-dependent gain (related to Gamma()) ---
    # As per the thesis, assume a linear gain from 2x at 0 sun to 1x at 1 sun (STC)
    # This prevents division by zero and keeps the gain factor bounded.
    low_light_gain = 2.0 - (irradiance / IRRADIANCE_STC)
    
    # --- Temperature performance gain (related to Upsilon()) ---
    # This factor represents the relative performance compared to Silicon
    si_temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    lle_temp_factor = 1 + BETA_LLE * (temperature - TEMP_STC)
    # The gain is the ratio of their performance factors
    temp_gain = lle_temp_factor / si_temp_factor
    
    # Calculate base power and apply gains
    base_power = P_RATED * (irradiance / IRRADIANCE_STC)
    enhanced_power = base_power * low_light_gain * temp_gain
    
    return np.maximum(0, enhanced_power)


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
        # ivcurve_pnts=100,
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

    self_consumption = pd.DataFrame({
        'SoC': SoC,
        'P_bat': P_bat,
        'P_available': P_available,
        'P_required': P_required,
        'P_h': P_h,
        'P_G2H': P_G2H,
        'P_H2G': P_H2G,
        'Bat_charge': Bat_charge,
        'Bat_discharge': Bat_discharge
    })

    return self_consumption

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

def calculate_total_energy(P_load, P_PV, self_consumption):
    total_energy = pd.DataFrame({
        'E_Load': (P_load/60)/1000,
        'E_PV': (P_PV/60)/1000,
        'E_G2H': (self_consumption['P_G2H']/60)/1000,
        'E_H2G': (self_consumption['P_H2G']/60)/1000,
        'E_charge': (self_consumption['Bat_charge']/60)/1000,
        'E_discharge': (self_consumption['Bat_discharge']/60)/1000,
    })

    return total_energy

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

    return pd.Series(interpolated_results)

def data_interpolation(total_energy_epv, total_energy_si):
    interpolated_total_energy = pd.DataFrame({
        'E_PV': interpolate_list(total_energy_si['E_PV'].sum(), total_energy_epv['E_PV'].sum()),
        'E_G2H': interpolate_list(total_energy_si['E_G2H'].sum(), total_energy_epv['E_G2H'].sum()),
        'E_H2G': interpolate_list(total_energy_si['E_H2G'].sum(), total_energy_epv['E_H2G'].sum()),
        'E_charge': interpolate_list(total_energy_si['E_charge'].sum(), total_energy_epv['E_charge'].sum()),
        'E_discharge': interpolate_list(total_energy_si['E_discharge'].sum(), total_energy_epv['E_discharge'].sum()),
    })

    return interpolated_total_energy

def data_per_season(total_energy_epv, total_energy_si, interpolated_values, SEASON_RANGES, DELTA_VALUES):
    E_PV_season[0] = total_energy_si['E_PV'].sum()
    E_G2H_season[0] = total_energy_si['E_G2H'].sum()
    E_H2G_season[0] = total_energy_si['E_H2G'].sum()
    E_discharge_season[0] = total_energy_si['E_discharge'].sum()
    E_charge_season[0] = total_energy_si['E_charge'].sum()

    for delta in range(4):
        E_PV_season[delta + 1] = interpolated_values['E_PV'][delta]
        E_G2H_season[delta + 1] = interpolated_values['E_G2H'][delta]
        E_H2G_season[delta + 1] = interpolated_values['E_H2G'][delta]
        E_discharge_season[delta + 1] = interpolated_values['E_discharge'][delta]
        E_charge_season[delta + 1] = interpolated_values['E_charge'][delta]

    E_PV_season[5] = total_energy_epv['E_PV'].sum()
    E_G2H_season[5] = total_energy_epv['E_G2H'].sum()
    E_H2G_season[5] = total_energy_epv['E_H2G'].sum()
    E_discharge_season[5] = total_energy_epv['E_discharge'].sum()
    E_charge_season[5] = total_energy_epv['E_charge'].sum()

    energy_pd = pd.DataFrame({
        'E_PV_season': E_PV_season,
        'E_G2H_season': E_G2H_season,
        'E_H2G_season': E_H2G_season,
        'E_discharge_season': E_discharge_season,
        'E_charge_season': E_charge_season,
    })
    print(E_charge_season)

    return energy_pd


if __name__ == "__main__":
    PV_data = PV_power_generation(irradiance['GHI'], temperature['t2m'], parameters, LLE_parameters)

    # Calculate self consumption for each technology
    self_consumption_si = calculate_self_consumption(PV_data['P_Si'])
    self_consumption_epv = calculate_self_consumption(PV_data['P_LLE'])

    # Plot both technologies
    stack_plot(P_load, self_consumption_si['P_G2H'], self_consumption_si['P_H2G'], self_consumption_si['Bat_charge'], self_consumption_si['Bat_discharge'], PV_data['P_Si'], 'Silicon')
    stack_plot(P_load, self_consumption_epv['P_G2H'], self_consumption_epv['P_H2G'], self_consumption_epv['Bat_charge'], self_consumption_epv['Bat_discharge'], PV_data['P_LLE'], 'EPV')
    # stack_plot(P_load, P_G2H_LLE, P_H2G_LLE, Bat_charge_LLE, Bat_discharge_LLE, PV_data['P_LLE'], 'LLE')

    # Calculate energy consumed
    total_energy_si = calculate_total_energy(P_load, PV_data['P_Si'], self_consumption_si)
    total_energy_epv = calculate_total_energy(P_load, PV_data['P_LLE'], self_consumption_epv)

    E_load = (P_load[winter_start:autumn_end].sum()/60)/1000 

    # interpolate values between known values
    interpolated_values = data_interpolation(total_energy_epv, total_energy_si)

    # Rewriting this part
    total_per_delta_mu = data_per_season(total_energy_epv, total_energy_si, interpolated_values, SEASON_RANGES, DELTA_VALUES)
    print(total_per_delta_mu)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.stackplot(DELTA_VALUES, (total_per_delta_mu['E_G2H_season']/E_load)*100, (np.abs(total_per_delta_mu['E_discharge_season'])/E_load)*100, (total_per_delta_mu['E_PV_season']/E_load)*100,
                  labels=['E_G2H', 'E_discharge', 'E_PV'],
                  colors=['#40679E', '#527853', '#FD841F'])
    plt.plot([1,2], [100,100], label='Load', color='black')

    plt.fill_between(DELTA_VALUES, [100,100,100,100,100,100], (total_per_delta_mu['E_charge_season']/E_load)*100, color='#AFC8AD', hatch='O', alpha =0.5, label='E_charge' )
    ax = plt.gca()  # Get the current Axes instance
    plt.xlim(1, 2)
    plt.ylim([0, 200])  # for example, to set the y-limit to 20% above max PV power
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


    # Splitting the data by season
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    split_indices = [1440, 2880, 4320, 5760]
    soc_si_seasons = np.split(SoC_Si, split_indices[:-1])  # Exclude the last index to match the number of seasons
    soc_lle_seasons = np.split(SoC_LLE, split_indices[:-1])

    # Preparing data for plotting
    data_to_plot_si = [season_data for season_data in soc_si_seasons]
    data_to_plot_lle = [season_data for season_data in soc_lle_seasons]
    data_to_plot = [val for pair in zip(data_to_plot_si, data_to_plot_lle) for val in pair]  # Interleave Si and LLE data
    labels = [f'{tech}\n{season}' for season in seasons for tech in ['Si', 'LLE']]


    # Define colors for the boxplots
    colors = ['blue', 'green']  # Grayscale colors for a professional look

    # Define the properties for the boxplot elements
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    whiskerprops = dict(linestyle='-', linewidth=1, color='black')
    capprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    flierprops = dict(marker='o', color='black', markersize=3)

    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate positions to add space between seasons
    n_groups = len(seasons)  # Number of seasons
    n_boxes_per_group = 2    # Number of box plots per group (Si and LLE)
    spacing = 1              # Space between groups
    positions = [i + (i // n_boxes_per_group) * spacing for i in range(n_groups * n_boxes_per_group)]

    # Creating the boxplot with the properties and custom positions
    bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, notch=True,
                       positions=positions,  # Use the custom positions
                       boxprops=boxprops, whiskerprops=whiskerprops,
                       capprops=capprops, medianprops=medianprops, flierprops=flierprops)

    # Coloring the boxes
    for patch, color in zip(bplot['boxes'], colors * (len(data_to_plot) // len(colors))):
        patch.set_facecolor(color)

    # Set font size and family for the plot
    plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

    # Setting axis labels and title
    ax.set_ylabel('SoC (%)')
    ax.set_xticklabels(labels, rotation=0)

    # Adjusting the y-axis limits
    ax.set_ylim(0, 100)
    ax = plt.gca()  # Get the current Axes instance
    # plt.xlim(1, 2)
    plt.ylim([0, 100])  # for example, to set the y-limit to 20% above max PV power
    for _, spine in ax.spines.items():
        spine.set_linewidth(1.5)

    # Increase the line width of the tick marks for visibility
    ax.tick_params(which='both', width=2)  # Applies to both major and minor ticks
    ax.tick_params(which='major', length=7)  # Only major ticks
    ax.tick_params(which='minor', length=4, color='gray')  # Only minor ticks

    # Add labels and title with a larger font size
    plt.xlabel('Season', fontsize=20)
    plt.show()
        
