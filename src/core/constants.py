# Physical and Characteristic Constants for the PINN Lithospheric Model

# Physical Constants
G_ACCEL = 9.81  # Gravity (m/s^2)
RHO_CRUST = 2700.0  # Baseline Crusaders Density (kg/m^3)
ETA_BASELINE = 1e21  # Baseline effective viscosity (Pa.s)
MU_BASELINE = 30e9  # Baseline shear modulus (Pa)

# Characteristic Scales (Non-Dimensionalization)
S0 = 1e8  # Characteristic Stress (Pa) - Typically 10-100 MPa
V0 = 1e-9  # Characteristic Velocity (m/s) - ~3 cm/year
L0 = 1e6  # Characteristic Length (m) - 1,000 km
T0 = L0 / V0  # Characteristic Time (s)
