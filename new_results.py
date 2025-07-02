import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters (Adjust these to fit your data) ---

# System and Standard Test Conditions (STC)
P_RATED = 1000  # Rated power of the system in Watts (e.g., 3 kW)
IRRADIANCE_STC = 1000  # Standard irradiance in W/m^2
TEMP_STC = 25  # Standard temperature in Celsius

# Temperature Coefficients (per degree Celsius)
# Typical for Silicon PV is -0.3% to -0.5% per degree C.
BETA_SI = -0.004
# Assume LLE-PV has a better (less negative) temperature coefficient
BETA_LLE = -0.0025


# --- 2. PV Power Models ---

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

def calculate_lle_power(irradiance, temperature):
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


# --- 3. Setup Grid and Calculate Power Difference ---

# Create arrays for irradiance and temperature ranges
irradiance_range = np.linspace(0, 1000, 100)  # 0 to 1000 W/m^2
temperature_range = np.linspace(0, 40, 100)   # 0 to 40 Celsius

# Create a 2D grid of values
irradiance_grid, temperature_grid = np.meshgrid(irradiance_range, temperature_range)

# Calculate power for each PV type across the entire grid
power_si = calculate_si_power(irradiance_grid, temperature_grid)
power_lle = calculate_lle_power(irradiance_grid, temperature_grid)

# Calculate the performance gain (the difference in power)
delta_p = power_lle - power_si

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

# Display the plot
plt.show()
