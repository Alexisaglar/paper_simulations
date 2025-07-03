from pvlib import pvsystem, temperature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PV_PARAMETERS
import time

# Constants
TEMPERATURE_STC = 25
# Temeprature coefficients
BETA_MU = -0.25
BETA_SI = -0.4
# EPV efficiency
MU_1SUN = 15
MU_0SUN = 30
# Silicon Efficiency
SI_1SUN = 20
SI_0SUN = 20


total_percentage_output_technology = []

def beta(temperature: np.array, technology:str = 'silicon') -> np.array:
    if technology == 'epv':
        beta = (BETA_MU - BETA_SI) * (temperature - TEMPERATURE_STC)
    else:
        beta = (BETA_SI - BETA_SI) * (temperature - TEMPERATURE_STC)
    return beta 

def phi(irradiance: np.array, technology:str ='silicon') -> np.array:
    if technology == 'epv':
        phi = MU_0SUN + ((MU_1SUN - MU_0SUN) / 1000) * irradiance
    else:
        phi = SI_0SUN + ((SI_1SUN - SI_0SUN) / 1000) * irradiance
    return phi

def delta_mu(irradiance, temperature, technology='silicon'):
    if technology == 'epv':
        # delta_mat = (beta(temperature) + phi(irradiance, technology))/ PV_PARAMETERS[f'pce_@1sun_{technology}']
        delta_mu = (beta(temperature, technology) + phi(irradiance, technology)) / MU_1SUN
    else:
        delta_mu = (beta(temperature, technology) + phi(irradiance, technology)) / SI_1SUN
    return delta_mu

def single_diode(irradiance, temperature):
    I_L, I_0, R_s, R_sh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance = irradiance,
        temp_cell = temperature,
        alpha_sc = PV_PARAMETERS['alpha_sc'],
        a_ref = PV_PARAMETERS['a_ref'],
        I_L_ref = PV_PARAMETERS['I_L_ref'],
        I_o_ref = PV_PARAMETERS['I_o_ref'],
        R_sh_ref = PV_PARAMETERS['R_sh_ref'],
        R_s = PV_PARAMETERS['R_s'],
        EgRef = PV_PARAMETERS['EgRef'],
        dEgdT = PV_PARAMETERS['dEgdT'],
    )
    curve_info = pvsystem.singlediode(
        photocurrent=I_L,
        saturation_current=I_0,
        resistance_series=R_s,
        resistance_shunt=R_sh,
        nNsVth=nNsVth,
        method='lambertw'
    )

    sd_output = curve_info['v_mp'] * curve_info['i_mp']

    return sd_output

def actual_pg(technology_output, season):
    # Calculate module power output with single diode diode model
    percentage_output_technology = []
    technology_df = pd.DataFrame(technology_output)

    for i, _ in enumerate(technology_output):
        # print(f'this is the total data im giving: {technology_output}')
        pg_percentage =  technology_output[i] / ((single_diode(1000, 25)* 15) / 10000)
        # plt.plot(pg_percentage)
        # print(f'this is the single value in the array: {pg_percentage}')
        percentage_output_technology.append(pg_percentage)
        print(f'this is the total appended in the array: {percentage_output_technology}')
        # plt.show()
    print(percentage_output_technology)
    plt.plot(np.arange(1, 25), percentage_output_technology[0], label='epv_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[1], label='epv_increased_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[2], label='sd_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[3], label='sd_percentage_increased_potential')
    plt.legend()
    plt.title(season)
    plt.show()
    return percentage_output_technology


def power_generation_pv(irradiance, temperature, season):
    # Calculate module power output with single diode diode model

    silicon_pv_power = single_diode(irradiance, temperature)
    pv_power_sd = np.array(silicon_pv_power * PV_PARAMETERS['series_cell'] * PV_PARAMETERS['parallel_cell']) / 10000
    pv_power_sd_decrease = pv_power_sd * delta_mat(irradiance, temperature, 'sd')

    # Calculating emerging PV possible power generation
    pv_power_epv = pv_power_sd * 0.75 * delta_mat(irradiance, temperature, 'epv')
    pv_power_epv_increased = pv_power_sd * delta_mat(irradiance, temperature, 'epv')

    # plotting pv output
    pv_power_plot([pv_power_epv, pv_power_epv_increased, pv_power_sd, pv_power_sd_decrease], ['epv', 'epv_increased', 'sd', 'sd_decreased'], season)

    # Calculating potential in % of energy output depending on season
    percentage_output_technology = actual_pg([pv_power_epv, pv_power_epv_increased, pv_power_sd, pv_power_sd_decrease], season)
    total_percentage_output_technology.append(percentage_output_technology)

    
    return pv_power_sd_decrease, pv_power_epv, pv_power_epv_increased, total_percentage_output_technology

def plot_heatmap(irradiance_grid: np.array, temperature_grid: np.array, pv_power: np.array, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    levels = np.linspace(np.min(pv_power), np.max(pv_power), 15)

    contour = ax.contourf(irradiance_grid, temperature_grid, pv_power, levels=levels, cmap='YlGn')

    cbar = fig.colorbar(contour)
    cbar.set_label('Power Difference (u_epv - u_si) in Watts', rotation=270, labelpad=20)

    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    ax.set_title(f'{name}', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'chapter_3_images/{name}.png', format='png', dpi=300, bbox_inches='tight')

    # plt.show()

def plot_3d(irradiance_grid: np.array, temperature_grid: np.array, pv_power: np.array, name: str) -> None:
    fig = plt.figure(figsize=(14, 10))
    
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(irradiance_grid, temperature_grid, pv_power,
                           cmap="YlGn", linewidth=0, antialiased=False)
    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    ax.set_title(f'{name}', pad=20, fontsize=16)
    ax.view_init(elev=30, azim=-60)
    plt.savefig(f'chapter_3_images/{name}.png', format='png', dpi=300, bbox_inches='tight')

    fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)

    # plt.show()


if __name__ == '__main__':
    irradiance_range = np.linspace(0, 1000, 100)
    temperature_range = np.linspace(0, 40, 100)

    temperature_grid, irradiance_grid = np.meshgrid(temperature_range, irradiance_range)

    power_si = np.ndarray(irradiance_grid.shape)
    power_epv = np.ndarray(irradiance_grid.shape)
    for i in range(irradiance_grid.shape[0]):
        for j in range(irradiance_grid.shape[1]):
            power_si[i, j] = single_diode(irradiance_range[i], temperature_range[j]) * (PV_PARAMETERS['series_cell'] * PV_PARAMETERS['parallel_cell'])
            power_epv[i, j] = power_si[i, j] * delta_mu(irradiance_range[i], temperature_range[j], 'epv')

    delta_p = power_epv - power_si

    plot_heatmap(irradiance_grid,  temperature_grid, delta_p, "PV generation difference heatmap")
    plot_heatmap(irradiance_grid,  temperature_grid, power_si, "pv generation Silicon PV heatmap")
    plot_heatmap(irradiance_grid,  temperature_grid, power_epv, "pv generation EPV")

    plot_3d(irradiance_grid,  temperature_grid, delta_p, "PV generation difference 3d")
    plot_3d(irradiance_grid,  temperature_grid, power_si, "PV Silicon 3d")
    plot_3d(irradiance_grid,  temperature_grid, power_epv, "PV EPV 3d")
