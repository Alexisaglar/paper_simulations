import numpy as np
#
# RESIDENTIAL_LOAD_FACTOR = np.array([
#     0.80, 0.73, 0.69, 0.64, 0.62, 0.61, 0.60, 0.61, 0.68, 0.77, 0.81, 0.83,
#     0.83, 0.84, 0.86, 0.84, 0.85, 0.84, 0.83, 0.81, 0.98, 1.00, 0.97, 0.90,
# ])
#
# INDUSTRIAL_LOAD_FACTOR = np.array([
#     0.30, 0.28, 0.24, 0.21, 0.20, 0.23, 0.30, 0.50, 0.54, 0.56, 0.58, 0.60, 
#     0.43, 0.40, 0.42, 0.80, 0.87, 0.96, 1.00, 0.97, 0.80, 0.53, 0.38, 0.34,
# ])
#
# COMMERCIAL_LOAD_FACTOR = np.array([
#     0.40, 0.38, 0.34, 0.32, 0.36, 0.47, 0.63, 0.84, 0.94, 1.00, 0.97, 0.88,
#     0.82, 0.60, 0.58, 0.56, 0.53, 0.52, 0.51, 0.48, 0.44, 0.49, 0.43, 0.42,
# ])
#

RESIDENTIAL_LOAD_FACTOR = np.array([
    0.30, 0.40, 0.44, 0.46, 0.50, 0.70, 0.72, 0.80, 0.70, 0.63, 0.50, 0.48,
    0.43, 0.50, 0.44, 0.55, 0.70, 0.85, 1.00, 0.85, 0.75, 0.65, 0.50, 0.44,
])

INDUSTRIAL_LOAD_FACTOR = np.array([
    0.65, 0.60, 0.65, 0.70, 0.80, 0.65, 0.65, 0.60, 0.60, 0.55, 0.50, 0.50, 
    0.50, 0.55, 0.60, 0.65, 0.60, 0.55, 0.68, 0.87, 0.90, 1.00, 0.90, 0.70,
])

COMMERCIAL_LOAD_FACTOR = np.array([
    0.40, 0.38, 0.34, 0.32, 0.36, 0.47, 0.63, 0.84, 0.94, 1.00, 0.97, 0.88,
    0.82, 0.80, 0.72, 0.73, 0.75, 0.65, 0.60, 0.52, 0.44, 0.49, 0.43, 0.42,
])

# SEASON FACTORS
WINTER_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

SUMMER_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

AUTUMN_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

SPRING_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

# ## Binary vector for loads positioning in the network
RESIDENTIAL_NODES = np.array([
    0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
    1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
    1, 0, 1
])
COMMERCIAL_NODES = np.array([
    0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 1, 0
])
INDUSTRIAL_NODES = np.array([
    1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0
])

PV_NODES = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0
])

# nodes with strings
NODE_TYPE = np.array([
    'industrial', 'residential',  'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'industrial', 'commercial',
    'residential', 'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'residential', 'commercial', 'commercial',
    'residential', 'residential', 'industrial', 'commercial', 'residential', 'industrial', 'commercial', 'industrial', 'commercial', 'residential',
    'residential', 'commercial', 'residential'
])

PEAK_LOAD = np.array([
        0.00, 0.10, 0.09, 0.12, 0.06, 0.06, 0.20, 0.20, 0.06, 0.06, 
        0.045, 0.06, 0.06, 0.12, 0.06, 0.06, 0.06, 0.09, 0.09,  0.09,
        0.09, 0.09, 0.09, 0.42, 0.42, 0.06, 0.06, 0.06, 0.12, 0.20,
        0.15, 0.21, 0.06,
]) 

# # # weather profile
# IRRADIANCE_EXAMPLE = np.array([
#     0, 0, 0, 0, 0, 50, 100, 200, 300, 350, 400, 550,
#     480, 400, 330, 280, 200, 150, 100, 50, 0, 0, 0, 0
# ])
#
# IRRADIANCE_EXAMPLE = np.array([0, 0, 0, 0, 0, 0, 0, 20, 50, 100, 150, 200, 150, 100, 50, 20, 0, 0, 0, 0, 0, 0, 0, 0])
IRRADIANCE_EXAMPLE = np.array([
    0, 0, 0, 0, 0, 50, 150, 270, 380, 450, 600, 650,
    700, 800, 700, 600, 400, 350, 200, 150, 0, 0, 0, 0
])
#
TEMPERATURE_EXAMPLE = np.full(24, 10)

# self-consumption script constants
SOC_MAX, SOC_MIN = 1.00, 0 # 1.0 and 0 because they are percentages
DISCHARGE_EFF, CHARGE_EFF = 0.95, 0.95
BATTERY_CAP = 0.5 # in kW
DELTA_TIME = 1
MAX_POWER_CHARGE, MAX_POWER_DISCHARGE = 2.50, 2.50

# PV PARAMETERS
PV_PARAMETERS = {
    'Name': 'SunPower SPR-305E-WHT-D',
    'BIPV': 'N',
    'Date': '10/5/2009',
    'T_NOCT': 42.4,
    'A_c': 1.7,
    'N_s': 96,
    'I_sc_ref': 5.96,
    'V_oc_ref': 64.2,
    'I_mp_ref': 5.58,
    'V_mp_ref': 54.7,
    'alpha_sc': 0.061745,
    'beta_oc': -0.2727,
    'a_ref': 2.3373,
    'I_L_ref': 5.9657,
    'I_o_ref': 6.3076e-12,
    'R_s': 0.37428,
    'R_sh_ref': 393.2054,
    'Adjust': 8.7,
    'gamma_r': -0.476,
    'series_cell': 5,
    'parallel_cell': 3,
    'Version': 'MM106',
    'EgRef': 1.121,
    'dEgdT': -0.0002677,
    'PTC': 200.1,
    'Technology': 'Mono-c-Si',
    'series_cell': 5,
    'parallel_cell': 3,
    'sd_t_c': -0.4,  # %/ºC
    'epv_t_c': -0.25,  # %/ºC
    'pce_@0sun_epv': 40,
    'pce_@1sun_epv': 20,
    'pce_@0sun_sd': 20,
    'pce_@1sun_sd': 20,
}
