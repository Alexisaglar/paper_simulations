import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters and Functions ---
# Using the same parameters as our previous visualizations
P_RATED = 3000  # Rated power of the system in Watts (e.g., 3 kW)
IRRADIANCE_STC = 1000  # Standard irradiance in W/m^2
TEMP_STC = 25  # Standard temperature in Celsius

# Temperature Coefficients (per degree Celsius)
BETA_SI = -0.004  # For the silicon model (μ₄)
BETA_LLE = -0.0025 # For the organic model (μ₂)

def calculate_si_power(irradiance, temperature):
    """Calculates power for the conventional Silicon PV model (μ₄)."""
    temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    power = P_RATED * (irradiance / IRRADIANCE_STC) * temp_factor
    return np.maximum(0, power)

def calculate_lle_power(irradiance, temperature):
    """Calculates power for the enhanced Organic PV model (μ₂)."""
    low_light_gain = 2.0 - (irradiance / IRRADIANCE_STC)
    si_temp_factor = 1 + BETA_SI * (temperature - TEMP_STC)
    lle_temp_factor = 1 + BETA_LLE * (temperature - TEMP_STC)
    temp_gain = lle_temp_factor / si_temp_factor
    base_power = P_RATED * (irradiance / IRRADIANCE_STC)
    enhanced_power = base_power * low_light_gain * temp_gain
    return np.maximum(0, enhanced_power)


# --- 2. Setup Grid and Calculate Power for Both Models ---
irradiance_range = np.linspace(0, 1000, 100)
temperature_range = np.linspace(0, 40, 100)
irradiance_grid, temperature_grid = np.meshgrid(irradiance_range, temperature_range)

# Calculate power across the grid for each model
power_si = calculate_si_power(irradiance_grid, temperature_grid)
power_lle = calculate_lle_power(irradiance_grid, temperature_grid)


# --- 3. Generate the Side-by-Side Heatmap Plot ---
# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Determine the shared min and max values for the color scale
vmin = min(power_si.min(), power_lle.min())
vmax = max(power_si.max(), power_lle.max())

# --- Subplot (a): Silicon PV (μ₄) ---
ax = axes[0]
contour_si = ax.contourf(irradiance_grid, temperature_grid, power_si,
                        levels=20, cmap='YlGn', vmin=vmin, vmax=vmax)
ax.set_title('(a) Silicon PV ($\mu_4$) Power Output', pad=15)
ax.set_xlabel('Irradiance (W/m²)')
ax.set_ylabel('Temperature (°C)')
ax.grid(True, linestyle='--', alpha=0.5)


# --- Subplot (b): Organic PV (μ₂) ---
ax = axes[1]
contour_lle = ax.contourf(irradiance_grid, temperature_grid, power_lle,
                         levels=20, cmap='YlGn', vmin=vmin, vmax=vmax)
ax.set_title('(b) Organic PV ($\mu_2$) Power Output', pad=15)
ax.set_xlabel('Irradiance (W/m²)')
# Y-axis label is shared due to sharey=True
ax.grid(True, linestyle='--', alpha=0.5)

# --- Add a single, shared color bar ---
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cbar = fig.colorbar(contour_lle, cax=cbar_ax)
cbar.set_label('Absolute Power Output (W)', rotation=270, labelpad=20)

# Add a main title for the entire figure
fig.suptitle('Comparison of Absolute Power Generation', fontsize=16)

# Adjust layout to prevent titles from overlapping
plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust rect to make space for suptitle and colorbar

plt.show()
