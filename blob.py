from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from parameters_pv import parameters, LLE_parameters
import time

# --- Your Original Constants and Setup ---
DELTA_VALUES = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
BETA_SI = 0.04
BETA_EPV = 0.025
TEMP_STC = 25
IRRADIANCE_STC = 1000
BATTERY_CHARGE_EFF = 0.95
BATTERY_DISCHARGE_EFF = 0.95
BATTERY_CHARGE_POWER_MAX = 2500
BATTERY_DISCHARGE_POWER_MAX = 2500
DELTA_TIME = 60
BATTERY_SOC_MAX = 100
BATTERY_SOC_MIN = 0
SEASON_RANGES_np = np.array(
    (np.array([0, 1439]),
    np.array([1440, 2879]),
    np.array([2880, 4319]),
    np.array([4320, 5759]),
    ))
SEASON_RANGES = {
    "Winter": [0, 1439], "Spring": [1440, 2879],
    "Summer": [2880, 4319], "Autumn": [4320, 5759]
}

# --- Your Original Data Loading ---
irradiance = pd.read_csv('data/irradiance_seasons.csv')
irradiance = np.array(irradiance['GHI'].to_numpy()).reshape(-1,1).squeeze()
temperature = pd.read_csv('data/temperature_seasons.csv')
temperature = np.array(temperature['t2m'].to_numpy()).reshape(-1,1).squeeze()
power_load_df = pd.read_csv('data/load_seasons.csv') * 1000
power_load = np.array(power_load_df[['winter', 'spring', 'summer', 'autumn']].to_numpy()).flatten(order='F')

colors = {
    'P_PV': '#FD841F',
    'P_G2H': '#40679E',
    'P_H2G': '#4CACBC',
    'Bat_charging': '#AFC8AD' ,
    'Bat_discharging': '#527853' ,
}

# --- Your Original Functions (with one modification) ---

def PV_power_generation(irradiance: np.array, temperature: np.array, parameters: dict, LLE_parameters: dict) -> pd.DataFrame:
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance=irradiance, temp_cell=temperature,
        alpha_sc=parameters['alpha_sc'], a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'], I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'], R_s=parameters['R_s'],
        EgRef=1.121, dEgdT=-0.0002677
    )
    curve_info = pvsystem.singlediode(
        photocurrent=IL, saturation_current=I0,
        resistance_series=Rs, resistance_shunt=Rsh,
        nNsVth=nNsVth, method='lambertw'
    )
    P_mp = curve_info['p_mp']
    PV_data = pd.DataFrame()
    PV_data['P_Si'] = P_mp * parameters['series_cell'] * parameters['parallel_cell']
    PV_data['irradiance'], PV_data['temperature'] = irradiance, temperature
    PV_data['phi'] = ((-1/10000) * irradiance) + LLE_parameters['PCE_ref']/100
    PV_data['delta'] = (PV_data['phi'] * 1.0) / (LLE_parameters['PCE_min']/100)
    PV_data['P_LLE'] = PV_data['P_Si'] * PV_data['delta']
    return PV_data.fillna(0)

# MODIFIED to accept bess_capacity_wh
def calculate_self_consumption(P_PV_np: np.array, power_load_np: np.array, bess_capacity_wh: float) -> pd.DataFrame:
    P_PV = pd.Series(P_PV_np)
    power_load = pd.Series(power_load_np)
    battery_soc = pd.Series(np.zeros(len(power_load)))
    power_battery = pd.Series(np.zeros(len(power_load)))

    if bess_capacity_wh == 0:
        power_grid = power_load - P_PV
        power_home_to_grid = -power_grid.clip(upper=0)
        power_grid_to_home = power_grid.clip(lower=0)
    else:
        for t in range(1, len(power_load)):
            if t in [s[0] for s in SEASON_RANGES.values() if s[0] > 0]:
                battery_soc.at[t-1] = 0
            power_mismatch = P_PV.at[t] - power_load.at[t]
            if power_mismatch < 0:
                power_available = (battery_soc.at[t-1] - BATTERY_SOC_MIN)/100 * bess_capacity_wh / (DELTA_TIME/60)
                discharge_power = min(-power_mismatch, power_available, BATTERY_DISCHARGE_POWER_MAX)
                power_battery.at[t] = -discharge_power
            else:
                power_required = (BATTERY_SOC_MAX - battery_soc.at[t-1])/100 * bess_capacity_wh / (DELTA_TIME/60)
                charge_power = min(power_mismatch, power_required, BATTERY_CHARGE_POWER_MAX)
                power_battery.at[t] = charge_power
            d_soc = (power_battery.at[t] * (DELTA_TIME/60) / bess_capacity_wh * 100)
            if power_battery.at[t] > 0: d_soc *= BATTERY_CHARGE_EFF
            elif power_battery.at[t] < 0: d_soc /= BATTERY_DISCHARGE_EFF
            battery_soc.at[t] = np.clip(battery_soc.at[t-1] + d_soc, BATTERY_SOC_MIN, BATTERY_SOC_MAX)
        power_grid = power_load - P_PV - power_battery
        power_home_to_grid = -power_grid.clip(upper=0)
        power_grid_to_home = power_grid.clip(lower=0)
        
    battery_charge = power_battery.clip(lower=0)
    battery_discharge = power_battery.clip(upper=0)
    return pd.DataFrame({'SoC': battery_soc, 'P_bat': power_battery, 'P_h': power_grid, 'P_G2H': power_grid_to_home, 'P_H2G': power_home_to_grid, 'Bat_charge': battery_charge, 'Bat_discharge': battery_discharge})

def calculate_total_energy(power_load, P_PV, self_consumption):
    time_step_h = DELTA_TIME / 60
    return pd.DataFrame({
        'E_Load': (power_load * time_step_h) / 1000, 'E_PV': (P_PV * time_step_h) / 1000,
        'E_G2H': (self_consumption['P_G2H'] * time_step_h) / 1000, 'E_H2G': (self_consumption['P_H2G'] * time_step_h) / 1000,
        'E_charge': (self_consumption['Bat_charge'] * time_step_h) / 1000, 'E_discharge': (self_consumption['Bat_discharge'] * time_step_h) / 1000,
    })

# --- ADDED PLOTTING FUNCTIONS ---

def create_bess_performance_plot(results_df, output_file='BESS_SSR_Line_Plot.png'):
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    axes = axes.flatten()
    tech_colors = {'Silicon': 'cornflowerblue', 'Organic': 'seagreen'}
    tech_labels = {'Silicon': r'Si-PV ($\mu_{si}$)', 'Organic': r'LLE-PV ($\mu_{epv}$)'}
    bess_capacities = sorted(results_df['BESS Capacity (kWh)'].unique())

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_df = results_df[results_df['Season'] == season]
        for tech in ['Silicon', 'Organic']:
            tech_df = season_df[season_df['PV Technology'] == tech]
            ax.plot(tech_df['BESS Capacity (kWh)'], tech_df['SSR'], marker='o', linestyle='-', label=tech_labels[tech], color=tech_colors[tech])
        ax.set_title(season, fontsize=16, fontweight='bold')
        if i % 2 == 0: ax.set_ylabel('Self-Sufficiency Ratio (SSR)', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(bess_capacities)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if i >= 2: ax.set_xlabel('BESS Capacity (kWh)', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=14)
    fig.suptitle('Seasonal Self-Sufficiency vs. BESS Capacity', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"Line plot saved as '{output_file}'")

def create_bess_performance_bar_chart(results_df, output_file='BESS_SSR_Bar_Chart.png'):
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    axes = axes.flatten()
    bar_width = 0.35
    bess_capacities = sorted(results_df['BESS Capacity (kWh)'].unique())
    x = np.arange(len(bess_capacities))
    tech_colors = {'Silicon': 'cornflowerblue', 'Organic': 'seagreen'}
    tech_labels = {'Silicon': r'Si-PV ($\mu_{si}$)', 'Organic': r'LLE-PV ($\mu_{epv}$)'}

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_df = results_df[results_df['Season'] == season]
        si_ssr = season_df[season_df['PV Technology'] == 'Silicon']['SSR']
        epv_ssr = season_df[season_df['PV Technology'] == 'Organic']['SSR']
        ax.bar(x - bar_width/2, si_ssr, bar_width, label=tech_labels['Silicon'], color=tech_colors['Silicon'])
        ax.bar(x + bar_width/2, epv_ssr, bar_width, label=tech_labels['Organic'], color=tech_colors['Organic'])
        ax.set_title(season, fontsize=16, fontweight='bold')
        if i % 2 == 0: ax.set_ylabel('Self-Sufficiency Ratio (SSR)', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c} kWh' for c in bess_capacities])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if i >= 2: ax.set_xlabel('BESS Capacity (kWh)', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=14)
    fig.suptitle('Seasonal Self-Sufficiency vs. BESS Capacity and PV Technology', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"Bar chart saved as '{output_file}'")

def create_bess_performance_heatmap(results_df, output_file='BESS_SSR_Heatmap.png'):
    si_df = results_df[results_df['PV Technology'] == 'Silicon'].pivot_table(index='Season', columns='BESS Capacity (kWh)', values='SSR')
    epv_df = results_df[results_df['PV Technology'] == 'Organic'].pivot_table(index='Season', columns='BESS Capacity (kWh)', values='SSR')
    diff_df = (epv_df - si_df).reindex(['Winter', 'Spring', 'Summer', 'Autumn'])

    plt.figure(figsize=(10, 7))
    sns.heatmap(diff_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, cbar_kws={'label': 'SSR Improvement (LLE-PV vs. Si-PV)'})
    plt.title(r'Performance Gain of LLE-PV ($\mu_{epv}$) over Si-PV ($\mu_{si}$)', fontsize=16, pad=20)
    plt.xlabel('BESS Capacity (kWh)', fontsize=12)
    plt.ylabel('Season', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap plot saved as '{output_file}'")

# --- NEW MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print("--- Starting Full BESS Analysis ---")
    
    # 1. Generate PV Power Data once
    print("Step 1: Calculating PV power generation...")
    PV_data = PV_power_generation(irradiance, temperature, parameters, LLE_parameters)
    
    # 2. Define Simulation Scenarios
    BESS_CAPACITIES_KWH = [0, 5, 10, 15]
    all_results_data = []

    # 3. Run Simulation Loop
    print("Step 2: Running simulations for all BESS capacities...")
    for capacity_kwh in BESS_CAPACITIES_KWH:
        capacity_wh = capacity_kwh * 1000
        print(f"  - Simulating for {capacity_kwh} kWh BESS...")
        
        # Run for both PV types
        for tech in ['Silicon', 'Organic']:
            pv_key = 'P_Si' if tech == 'Silicon' else 'P_LLE'
            
            sc = calculate_self_consumption(PV_data[pv_key].to_numpy(), power_load, bess_capacity_wh=capacity_wh)
            te = calculate_total_energy(power_load, PV_data[pv_key].to_numpy(), sc)
            
            # Calculate seasonal SSR
            for season_name, (start, end) in SEASON_RANGES.items():
                te_season = te.loc[start:end]
                E_load_season = te_season['E_Load'].sum()
                E_g2h_season = te_season['E_G2H'].sum()
                ssr = (E_load_season - E_g2h_season) / E_load_season if E_load_season > 0 else 0
                
                all_results_data.append({
                    'BESS Capacity (kWh)': capacity_kwh,
                    'Season': season_name,
                    'PV Technology': tech,
                    'SSR': ssr
                })

    # 4. Create final DataFrame
    results_df = pd.DataFrame(all_results_data)
    results_df.to_csv('seasonal_summary_by_bess.csv', index=False, float_format='%.3f')
    print("\nStep 3: Seasonal summary data saved to 'seasonal_summary_by_bess.csv'")

    # 5. Generate all three plots
    print("\nStep 4: Generating plots...")
    create_bess_performance_plot(results_df)
    create_bess_performance_bar_chart(results_df)
    create_bess_performance_heatmap(results_df)

    print("\n--- Analysis Complete ---")
