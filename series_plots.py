from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parameters_pv import parameters, LLE_parameters
import time

DELTA_VALUES = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])  # The scaling parameter values

BETA_SI = 0.04
BETA_EPV = 0.025
TEMP_STC = 25
IRRADIANCE_STC = 1000

# Constants
BATTERY_CAPACITY= 5000  # Battery capacity in kWh, for example
BATTERY_CHARGE_EFF = 0.95  # Charging efficiency
BATTERY_DISCHARGE_EFF = 0.95  # Discharging efficiency
BATTERY_CHARGE_POWER_MAX = 2500  # Charging power max
BATTERY_DISCHARGE_POWER_MAX = 2500  # Discharging power max
DELTA_TIME = 60  # Time step in hours
BATTERY_SOC_MAX = 100  # 100%
BATTERY_SOC_MIN = 0  # 0%

SEASON_RANGES = np.array(
    (np.array([0, 1439]),
    np.array([1440, 2879]),
    np.array([2880, 4319]),
    np.array([4320, 5759]),
    ))


irradiance = pd.read_csv('data/irradiance_seasons.csv')
irradiance = np.array(irradiance['GHI'].to_numpy()).reshape(-1,1).squeeze()
temperature = pd.read_csv('data/temperature_seasons.csv')
temperature = np.array(temperature['t2m'].to_numpy()).reshape(-1,1).squeeze()
power_load = pd.read_csv('data/load_seasons.csv') * 1000
power_load = np.array(power_load[['winter', 'spring', 'summer', 'autumn']].to_numpy()).flatten(order='F')

energy_pv_season = pd.Series(len(DELTA_VALUES))
energy_g2h_season = pd.Series(len(DELTA_VALUES))
energy_h2g_season = pd.Series(len(DELTA_VALUES))
energy_discharge_season = pd.Series(len(DELTA_VALUES))
energy_charge_season = pd.Series(len(DELTA_VALUES))
energy_season_total = np.zeros((len(SEASON_RANGES), 5, len(DELTA_VALUES)))

# color scheme
colors = {
    'P_PV': '#FD841F',    # Orange for solar PV generation
    'P_G2H': '#40679E',   # Green for generation from Grid to Hydrogen
    'P_H2G': '#4CACBC',   # Cyan for Hydrogen to Grid
    'Bat_charging': '#AFC8AD' ,   # Yellow for battery storage
    'Bat_discharging': '#527853' ,   # Yellow for battery storage
}

def PV_power_generation(irradiance: np.array, temperature: np.array, parameters: np.array, LLE_parameters: np.array) -> pd.DataFrame:
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

def calculate_self_consumption(P_PV: np.array) -> pd.DataFrame:
    # Initialize variables
    battery_soc = pd.Series(np.zeros(len(power_load)))  # State of Charge
    power_battery = pd.Series(np.zeros(len(power_load)))  # Battery power
    for t in range(1, len(power_load)):
        power_available = ((BATTERY_SOC_MIN - battery_soc[t-1]/100)) * (BATTERY_CAPACITY * DELTA_TIME) * BATTERY_DISCHARGE_EFF# Power available
        power_required = ((((BATTERY_SOC_MAX - battery_soc[t-1])/100) * (BATTERY_CAPACITY * DELTA_TIME)) / (BATTERY_CHARGE_EFF))  # Power required

        if t == 1440 or t == 2880 or t == 4320 or t == 0: 
            battery_soc[t-1] = 0 

        if power_load[t] > P_PV[t]:
            power_battery[t] = max(P_PV[t] - power_load[t], power_available, -BATTERY_DISCHARGE_POWER_MAX)
        # elif power_load.iloc[t] < P_PV.iloc[t]:
        else:
            power_battery[t] = min(P_PV[t] - power_load[t], power_required, BATTERY_CHARGE_POWER_MAX)
            # P_bat[t] = 0

        Z_bat = 1 if power_battery[t] > 0 else 0
        if Z_bat == 1 :
            battery_soc[t] = (battery_soc[t-1] + (BATTERY_CHARGE_EFF * ((power_battery[t] / DELTA_TIME) / BATTERY_CAPACITY) * 100))
        else:
            battery_soc[t] = battery_soc[t-1] + (((power_battery[t] / DELTA_TIME * BATTERY_DISCHARGE_EFF ) / BATTERY_CAPACITY )* 100) 

    # Calculate P_h^t , P_H2G, P_G2H
    power_grid = power_load - P_PV + power_battery  

    # Divide P_h in P_H2G and P_G2H
    power_home_to_grid = pd.Series(np.zeros(len(power_grid)))  # Battery power
    power_grid_to_home = pd.Series(np.zeros(len(power_grid)))  # Battery power
    for i in range(len(power_grid)):
        if power_grid[i] > 0:
            power_grid_to_home[i] = power_grid[i] 
        else:
            power_home_to_grid[i] = power_grid[i] 

    # Divide in bat_charge and discharge for better graphing
    battery_charge = pd.Series(np.zeros(len(power_battery)))  # Battery power
    battery_discharge = pd.Series(np.zeros(len(power_battery)))  # Battery power
    for i in range(len(power_battery)):
        if power_battery[i] > 0:
            battery_charge[i] = power_battery[i] 
        else:
            battery_discharge[i] = power_battery[i] 

    self_consumption = pd.DataFrame({
        'SoC': battery_soc,
        'P_bat': power_battery,
        'P_available': power_available,
        'P_required': power_required,
        'P_h': power_grid,
        'P_G2H': power_grid_to_home,
        'P_H2G': power_home_to_grid,
        'Bat_charge': battery_charge,
        'Bat_discharge': battery_discharge
    })
    
    return self_consumption

def stack_plot(
    power_load: np.array,
    P_G2H: pd.Series,
    P_H2G: pd.Series,
    Bat_charge: pd.Series,
    Bat_discharge: pd.Series,
    P_PV: pd.Series,
    name: str
) -> None:
    # Plot figure
    plt.plot(power_load)
    plt.show()
    plt.figure(figsize=(12, 6))
    # Increase the overall font sizes for readability
    plt.rcParams.update({'font.size': 20})  # Adjust as needed for IEEE guidelines
    plt.stackplot(range(len(power_load)), 
                (power_load - P_G2H)/1000,
                P_G2H/1000, 
                abs(P_H2G)/1000, 
                Bat_charge/1000, 
                Bat_discharge/1000, 
                labels=['PV to Load','Grid to Home', 'Home to Grid', 'Battery Charging', 'Battery Discharging'], 
                colors=colors.values(),
                alpha=0.8)

    plt.plot(range(len(power_load)), 
            power_load/1000, label='Load Power', 
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

def calculate_total_energy(power_load, P_PV, self_consumption):
    total_energy = pd.DataFrame({
        'E_Load': (power_load/60)/1000,
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
        'E_PV': interpolate_list(total_energy_epv['E_PV'].sum(), total_energy_si['E_PV'].sum()),
        'E_G2H': interpolate_list(total_energy_epv['E_G2H'].sum(), total_energy_si['E_G2H'].sum()),
        'E_H2G': interpolate_list(total_energy_epv['E_H2G'].sum(), total_energy_si['E_H2G'].sum()),
        'E_charge': interpolate_list(total_energy_epv['E_charge'].sum(), total_energy_si['E_charge'].sum()),
        'E_discharge': interpolate_list(total_energy_epv['E_discharge'].sum(), total_energy_si['E_discharge'].sum()),
    })

    return interpolated_total_energy

def data_per_season(
    total_energy_epv: pd.DataFrame,
    total_energy_si: pd.DataFrame,
    # interpolated_values: pd.DataFrame,
    SEASON_RANGES: np.array,
    DELTA_VALUES: np.array
) -> pd.DataFrame:

    columns = ['E_PV', 'E_G2H', 'E_H2G', 'E_discharge', 'E_charge']
    num_seasons = len(SEASON_RANGES)
    num_energy_types = len(columns)
    num_scenarios = len(DELTA_VALUES)

    energy_season_total = np.zeros((num_seasons, num_scenarios, num_energy_types))

    for season_idx in range(num_seasons):
        season_start = SEASON_RANGES[season_idx, 0]
        season_end = SEASON_RANGES[season_idx, 1]

        energy_season_si = total_energy_si.loc[season_start:season_end, columns].sum()
        energy_season_epv = total_energy_epv.loc[season_start:season_end, columns].sum()
        
        energy_interpolated = np.array(data_interpolation(energy_season_si, energy_season_epv)).reshape(4,5).squeeze().T

        si_column = energy_season_si.to_numpy().reshape(-1,1)
        epv_column = energy_season_epv.to_numpy().reshape(-1,1)

        print(energy_interpolated)
        combined_season_data = np.concatenate(
            [si_column, energy_interpolated, epv_column], axis=1
        )
        
        # Assign the entire (5, 6) block to the current season's slice.
        energy_season_total[season_idx, :, :] = combined_season_data.T

    return energy_season_total

def plot_seasonal_stack(
    season_index: int, 
    season_name: str, 
    seasonal_data: np.ndarray, 
    delta_values: np.ndarray, 
    energy_load: float
) -> None:
    # 0:E_PV, 1:E_G2H, 2:E_H2G, 3:E_discharge, 4:E_charge
    e_pv_data = seasonal_data[season_index, :, 0]
    e_g2h_data = seasonal_data[season_index, :, 1]
    e_discharge_data = seasonal_data[season_index, :, 3]

    e_pv_perc = (e_pv_data / energy_load) * 100
    e_g2h_perc = (e_g2h_data / energy_load) * 100
    e_discharge_perc = (np.abs(e_discharge_data) / energy_load) * 100

    plt.figure(figsize=(10, 6))
    plt.stackplot(
        delta_values, 
        e_pv_perc, 
        e_g2h_perc, 
        e_discharge_perc,
        labels=['E_PV', 'E_G2H', 'E_discharge'],
        colors=['#FD841F', '#40679E', '#527853'] # Match colors to data order
    )

    # Use axhline for a clean horizontal line across the plot.
    plt.axhline(100, color='black', linestyle='--', label='Load')
    
    plt.title(f'Energy Contribution for {season_name}')
    plt.xlabel('Scenario (from SI to EPV)')
    plt.ylabel('Energy Contribution (% of Load)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set plot limits for better visualization
    plt.xlim(delta_values.min(), delta_values.max())
    plt.ylim(0, max(110, plt.ylim()[1])) # Ensure the 100% line is visible

    plt.show()

if __name__ == "__main__":
    PV_data = PV_power_generation(irradiance, temperature, parameters, LLE_parameters)

    # Calculate self consumption for each technology
    self_consumption_si = calculate_self_consumption(PV_data['P_Si'])
    self_consumption_epv = calculate_self_consumption(PV_data['P_LLE'])

    # Plot both technologies
    stack_plot(power_load, self_consumption_si['P_G2H'], self_consumption_si['P_H2G'], self_consumption_si['Bat_charge'], self_consumption_si['Bat_discharge'], PV_data['P_Si'], 'Silicon')
    stack_plot(power_load, self_consumption_epv['P_G2H'], self_consumption_epv['P_H2G'], self_consumption_epv['Bat_charge'], self_consumption_epv['Bat_discharge'], PV_data['P_LLE'], 'EPV')
    # stack_plot(power_load, P_G2H_LLE, P_H2G_LLE, Bat_charge_LLE, Bat_discharge_LLE, PV_data['P_LLE'], 'LLE')

    # Calculate energy consumed
    total_energy_si = calculate_total_energy(power_load, PV_data['P_Si'], self_consumption_si)
    total_energy_epv = calculate_total_energy(power_load, PV_data['P_LLE'], self_consumption_epv)

    energy_load = (power_load.sum()/60)/1000 

    # # interpolate values between known values
    # interpolated_values = data_interpolation(total_energy_epv, total_energy_si)

    # Rewriting this part
    total_per_delta_mu = data_per_season(total_energy_epv, total_energy_si, SEASON_RANGES, DELTA_VALUES)
    print('this is the details')
    print(total_per_delta_mu[0,:,:])
    season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
    for i, name in enumerate(season_names):
        print(f"Generating plot for {name}...")
        plot_seasonal_stack(
            season_index=i,
            season_name=name,
            seasonal_data=total_per_delta_mu,
            delta_values=DELTA_VALUES,
            energy_load=energy_load # Assuming load might vary by season
        )

    plt.stackplot(DELTA_VALUES, (total_per_delta_mu[0,:,0]/energy_load)*100,
                  labels=['E_G2H', 'E_discharge', 'E_PV'],
                  colors=['#40679E', '#527853', '#FD841F'])

    plt.show()
    # Plot
    plt.figure(figsize=(10, 6))
    plt.stackplot(DELTA_VALUES, (total_per_delta_mu[0,:,:]/energy_load)*100, (np.abs(total_per_delta_mu[0,:,:])/energy_load)*100, (total_per_delta_mu[0,:,:]/energy_load)*100,
                  labels=['E_G2H', 'E_discharge', 'E_PV'],
                  colors=['#40679E', '#527853', '#FD841F'])
    plt.plot([1,2], [100,100], label='Load', color='black')

    plt.fill_between(DELTA_VALUES, [100,100,100,100,100,100], (total_per_delta_mu['E_charge_season']/energy_load)*100, color='#AFC8AD', hatch='O', alpha =0.5, label='E_charge' )
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
    soc_si_seasons = np.split(self_consumption_si['SoC'], split_indices[:-1])  # Exclude the last index to match the number of seasons
    soc_lle_seasons = np.split(self_consumption_epv['SoC'], split_indices[:-1])

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
        
