from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PV_PARAMETERS
import time

total_percentage_output_technology = []

def beta(temperature):
    beta = (PV_PARAMETERS['sd_t_c'] - PV_PARAMETERS['epv_t_c']) * temperature 
    return beta 
    
def phi(irradiance, technology):
    phi = (PV_PARAMETERS[f'pce_@0sun_{technology}'] + ((PV_PARAMETERS[f'pce_@1sun_{technology}'] - PV_PARAMETERS[f'pce_@0sun_{technology}']) / 1000 ) * irradiance)
    return phi

def delta_mat(irradiance, temperature, technology):
    delta_mat = (beta(temperature) + phi(irradiance, technology))/ PV_PARAMETERS[f'pce_@1sun_{technology}']
    # print(delta_mat)
    return delta_mat

def pv_power_plot(data, label, season):
    for i, _ in enumerate(data):
        plt.plot(data[i], label=label[i])
        plt.title(f'power {season}')
    plt.legend()
    plt.savefig("pv_gen.png")
    plt.show()

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


if __name__ == '__main__':
    irradiance_range = np.linspace(0, 1000, 100)
    temperature_range = np.linspace(0, 40, 100)

    irradiance_grid, temperature_grid = np.meshgrid(irradiance_range, temperature_range)
    # print(irradiance_grid)
    # print(temperature_grid)
    # print(irradiance_grid.shape[0])

    power_si = np.ndarray(irradiance_grid.shape)
    power_epv = np.ndarray(irradiance_grid.shape)
    for i in range(irradiance_grid.shape[0]):
        for j in range(irradiance_grid.shape[1]):
            power_si[i, j] = single_diode(irradiance_range[i], temperature_range[j]) * (PV_PARAMETERS['series_cell'] * PV_PARAMETERS['parallel_cell']) / 10000
            power_epv[i, j] = power_si[i, j] * delta_mat(irradiance_range[i], temperature_range[j], 'epv')

    delta_p = power_epv - power_si

    # --- Generate the "Difference Surface" 3D Plot ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the difference (delta_p) as a single surface
    # The colormap is tied to the height (z-value) of the gain
    surf = ax.plot_surface(irradiance_grid, temperature_grid, power_si,
                           cmap="YlGn", linewidth=0, antialiased=False)

    # --- Customize the Plot ---
    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    ax.set_zlabel('Power Gain (ΔP) in Watts', labelpad=10)
    ax.set_title('3D Power Gain Surface of LLE-PV over Si-PV', pad=20, fontsize=16)

    # Set viewing angle
    ax.view_init(elev=30, azim=-60)

    # Add a color bar that maps to the z-values (the power gain)
    fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)

    plt.show()




    # --- 4. Generate the Contour Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the filled contour plot
    # Levels are set to create clear boundaries in the plot
    levels = np.linspace(np.min(delta_p), np.max(delta_p), 15)
    contour = ax.contourf(irradiance_grid, temperature_grid, delta_p, levels=levels, cmap='YlGn')

    # Add a color bar
    cbar = fig.colorbar(contour)
    cbar.set_label('Power Gain (deltaP) in Watts', rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel('Irradiance (W/m^2)')
    ax.set_ylabel('Temperature (degreeC)')
    ax.set_title('Performance Gain Map: LLE-PV vs. Si-PV', pad=20)

    # Show grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- Generate the "Difference Surface" 3D Plot ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the difference (delta_p) as a single surface
    # The colormap is tied to the height (z-value) of the gain
    surf = ax.plot_surface(irradiance_grid, temperature_grid, delta_p,
                           cmap="YlGn", linewidth=0, antialiased=False)

    # --- Customize the Plot ---
    ax.set_xlabel('Irradiance (W/m²)', labelpad=15)
    ax.set_ylabel('Temperature (°C)', labelpad=15)
    ax.set_zlabel('Power Gain (ΔP) in Watts', labelpad=10)
    ax.set_title('3D Power Gain Surface of LLE-PV over Si-PV', pad=20, fontsize=16)

    # Set viewing angle
    ax.view_init(elev=30, azim=-60)

    # Add a color bar that maps to the z-values (the power gain)
    fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)

    plt.show()


    # irradiance = np.arange(0, 1100, 100)
    # temperature = np.full(11, 25)
    # temperature_seasons = np.load('data/season_temperature.npy')
    # irradiance_seasons = np.load('data/season_irradiance.npy')
    # seasons = ['autumn', 'spring', 'summer', 'winter']

    # for i in range(irradiance_seasons.shape[1]):
    #     pv_power_sd, pv_power_epv, pv_power_epv_increased, total_potential_energy = power_generation_pv(irradiance_seasons[:, i], temperature_seasons[:, i], seasons[i])
    # plt.plot(irradiance_grid, temperature_grid, power_si)
    # plt.show()
    # plt.plot(irradiance_grid, temperature_grid, power_si)
    # plt.show()
    #     # plot_power_output(pv_power_sd, pv_power_epv)
