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

SERIES_CELLS = 5
PARALLEL_CELLS = 3

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

def plot_heatmap(irradiance_grid: np.array, temperature_grid: np.array, pv_power: np.array, name: str, cmap: str) -> None:
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
    })

    fig, ax = plt.subplots(figsize=(10, 4))

    levels = np.linspace(np.min(pv_power), np.max(pv_power), 15)
    levels = np.round(levels, 2)

    contour = ax.contourf(irradiance_grid, temperature_grid, pv_power, levels=levels, cmap=cmap)

    cbar = fig.colorbar(contour)
    cbar.set_label('Power Difference (kW)', rotation=270, labelpad=30)

    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    # ax.set_title(f'{name}', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'chapter_3_images/{name}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_3d(irradiance_grid: np.array, temperature_grid: np.array, pv_power: np.array, name: str, cmap: str) -> None:
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
    })
    fig = plt.figure(figsize=(14, 10))
    
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(irradiance_grid, temperature_grid, pv_power,
                           cmap=cmap, linewidth=0, antialiased=False)
    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    # ax.set_title(f'{name}', pad=20, fontsize=16)
    ax.view_init(elev=5, azim=-40)
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Power Difference (kW)', rotation=270, labelpad=30)
    plt.savefig(f'chapter_3_images/{name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    irradiance_range = np.linspace(0, 1000, 100)
    temperature_range = np.linspace(0, 40, 100)

    temperature_grid, irradiance_grid = np.meshgrid(temperature_range, irradiance_range)

    power_si = np.ndarray(irradiance_grid.shape)
    power_epv = np.ndarray(irradiance_grid.shape)
    for i in range(irradiance_grid.shape[0]):
        for j in range(irradiance_grid.shape[1]):
            power_si[i, j] = single_diode(irradiance_range[i], temperature_range[j]) * SERIES_CELLS * PARALLEL_CELLS
            power_epv[i, j] = power_si[i, j] * delta_mu(irradiance_range[i], temperature_range[j], 'epv')

    power_epv = power_epv/1000
    power_si = power_si/1000
    delta_p = power_epv - power_si

    plot_heatmap(irradiance_grid,  temperature_grid, delta_p, "PV generation difference heatmap", 'YlOrBr')
    plot_heatmap(irradiance_grid,  temperature_grid, power_si, "pv generation Silicon PV heatmap", 'YlGn')
    plot_heatmap(irradiance_grid,  temperature_grid, power_epv, "pv generation EPV", 'YlGn')

    plot_3d(irradiance_grid,  temperature_grid, delta_p, "PV generation difference 3d", 'YlOrBr')
