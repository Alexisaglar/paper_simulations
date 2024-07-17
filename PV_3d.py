import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import pvsystem
from datetime import datetime
from parameters_pv import parameters
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize


series_panel = 5
parallel_panel = 3
# CFPV Data
PCE_ref_CFPV = 10
# y=mx+b
slope_2x_enhance = -1 / 100
constant_2x_enhance = 20
irradiance = np.linspace(0, 1000, 50)  # From 0 to 1 sun (1000 W/m^2)
temperature = np.linspace(0, 25, 50)  # Temperature range


def pv_generation(
    irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV, parameters
):
    # Preallocate the output array
    P_out = np.zeros((len(irradiance), len(temperature)))

    # Loop over each combination of irradiance and temperature
    for i, G in enumerate(irradiance):
        for j, T in enumerate(temperature):
            IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                effective_irradiance=G,
                temp_cell=T,
                alpha_sc=parameters["alpha_sc"],
                a_ref=parameters["a_ref"],
                I_L_ref=parameters["I_L_ref"],
                I_o_ref=parameters["I_o_ref"],
                R_sh_ref=parameters["R_sh_ref"],
                R_s=parameters["R_s"],
                EgRef=1.121,
                dEgdT=-0.0002677,
            )

            curve_info = pvsystem.singlediode(
                photocurrent=IL,
                saturation_current=I0,
                resistance_series=Rs,
                resistance_shunt=Rsh,
                nNsVth=nNsVth,
                # ivcurve_pnts=100,
                method="lambertw",
            )

            # Calculate the module power
            P_mp = curve_info["i_mp"] * curve_info["v_mp"]

            # Scale by the number of series and parallel panels
            P_out[i, j] = P_mp * series_panel * parallel_panel

    return P_out


# Generate a 2D array of power outputs
P = pv_generation(
    irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV, parameters
)

# Calculate other performance metrics over the grid
PCE_at_GHI = slope_2x_enhance * irradiance[:, np.newaxis] + constant_2x_enhance
P_PCE_at_GHI = np.full_like(P, 20)
P_CFPV = P * (PCE_at_GHI / PCE_ref_CFPV)

# Create meshgrids for plotting
T, I = np.meshgrid(temperature, irradiance)

# Plotting the 3D surfaces
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot for the original technology
surf1 = ax.plot_wireframe(
    T, I, P/1000, cmap="YlGnBu", rstride=1, cstride=1, alpha=0.8, label="Original Tech"
)

# Plot for the enhanced technology
surf2 = ax.plot_surface(
    T, I, P_CFPV/1000, cmap="YlGnBu", rstride=1, cstride=1, alpha=0.8, label="Enhanced Tech"
)

# Create a ScalarMappable with the colormap and normalization range
norm = Normalize(vmin=1, vmax=2)
sm = plt.cm.ScalarMappable(cmap="YlGnBu_r", norm=norm)
sm.set_array([])  # Set the array for the ScalarMappable

# Create the color bar with the updated ScalarMappable
cbar = fig.colorbar(sm, shrink=0.5, aspect=5, ax=ax)

# Update the color bar ticks and labels to reflect the new range
cbar.set_ticks([2, 1.75, 1.5, 1.25, 1])
cbar.set_ticklabels(["2", "1.75", "1.5", "1.25", "1"])
cbar.set_label("δ_mat")


ax.set_ylabel("Irradiance (W/m²)", fontsize=12, labelpad=1)
ax.set_xlabel("Temperature (°C)", fontsize=12, labelpad=1)
ax.set_zlabel("Power Output (kW)", fontsize=12, labelpad=1)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set gridlines
ax.xaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.5)
ax.yaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.5)
ax.zaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.5)


# # Hide the tick labels
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

fig.savefig("3d_plot.png", dpi=300, bbox_inches="tight")
plt.show()
