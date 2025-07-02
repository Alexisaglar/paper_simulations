import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- Reusing the same models and parameters ---
P_RATED = 3000
IRRADIANCE_STC = 1000
TEMP_STC = 25
BETA_SI = -0.004
BETA_LLE = -0.0025

def calculate_si_power(irradiance, temperature):
    temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    power = P_RATED * (irradiance / IRRADIANCE_STC) * temp_factor
    return np.maximum(0, power)

def calculate_lle_power(irradiance, temperature):
    low_light_gain = 2.0 - (irradiance / IRRADIANCE_STC)
    si_temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    lle_temp_factor = 1 + BETA_LLE * (temperature - TEMP_STC)
    temp_gain = lle_temp_factor / si_temp_factor
    base_power = P_RATED * (irradiance / IRRADIANCE_STC)
    enhanced_power = base_power * low_light_gain * temp_gain
    return np.maximum(0, enhanced_power)

# --- Setup Grid and Calculate Power Difference ---
irradiance_range = np.linspace(0, 1000, 50)
temperature_range = np.linspace(0, 40, 50)
irradiance_grid, temperature_grid = np.meshgrid(irradiance_range, temperature_range)
power_si = calculate_si_power(irradiance_grid, temperature_grid)
power_lle = calculate_lle_power(irradiance_grid, temperature_grid)
delta_p = power_lle - power_si


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
