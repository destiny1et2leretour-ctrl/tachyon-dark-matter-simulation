# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:42:36 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m0 = 1.0        # Initial imaginary mass (m0^2)
lam = 0.25      # Self-interaction constant (lambda)
Lambda_DE = 0.1 # Desired residual Dark Energy value (Lambda)

# Offset calculation so that the bottom of the potential (V_min) equals Lambda_DE
# Theoretical minimum: v = sqrt(m0^2 / 4*lambda)
offset = Lambda_DE + (m0**4) / (16 * lam)

def potential(phi):
    """Equation (2) from Book 1: The Mexican hat"""
    return -0.5 * m0**2 * phi**2 + lam * phi**4 + offset

def potential_derivative(phi):
    """Force exerted on the field (dV/dphi)"""
    return -m0**2 * phi + 4 * lam * phi**3


# ==========================================
# The Klein-Gordon equation in an expanding universe simplifies to a 
# damped oscillator: d2phi/dt2 + 3H(dphi/dt) + dV/dphi = 0
friction = 0.6  # Represents the expansion friction (3H)

def equation_of_motion(t, y):
    phi, dphi_dt = y
    # y[0] is phi, y[1] is the velocity of phi
    d2phi_dt2 = -friction * dphi_dt - potential_derivative(phi)
    return [dphi_dt, d2phi_dt2]


# The tachyon starts almost at 0, pushed by a tiny quantum fluctuation
y0 = [0.01, 0.0]  # [initial phi, initial velocity]
t_span = (0, 30)  # Simulated time
t_eval = np.linspace(t_span[0], t_span[1], 500)

solution = solve_ivp(equation_of_motion, t_span, y0, t_eval=t_eval)

phi_t = solution.y[0]
velocity_phi_t = solution.y[1]

# Calculation of Energy Density (rho) and Pressure (P)
rho_t = 0.5 * velocity_phi_t**2 + potential(phi_t)
P_t = 0.5 * velocity_phi_t**2 - potential(phi_t)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Plot 1: The fall into the Mexican Hat ----
phi_range = np.linspace(-1.5, 1.5, 200)
ax1.plot(phi_range, potential(phi_range), 'k-', label='Potential $V(\phi)$')
ax1.plot(phi_t, potential(phi_t), 'r.', markersize=2, alpha=0.5, label='Tachyon Trajectory')
ax1.plot(phi_t[0], potential(phi_t[0]), 'go', markersize=8, label='False Vacuum (Big Bang)')
ax1.plot(phi_t[-1], potential(phi_t[-1]), 'bo', markersize=8, label='True Vacuum (Dark Energy)')

ax1.set_title("Symmetry Breaking (False Vacuum)")
ax1.set_xlabel("Tachyon field value $\phi$")
ax1.set_ylabel("Potential Energy $V(\phi)$")
ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---- Plot 2: Density and Pressure (Emergence of Dark Energy) ----
ax2.plot(solution.t, rho_t, 'b-', linewidth=2, label='Energy Density ($\\rho$)')
ax2.plot(solution.t, P_t, 'r-', linewidth=2, label='Pressure ($P$)')

ax2.set_title("Cosmic Evolution: Creation of Dark Energy")
ax2.set_xlabel("Cosmic time ($t$)")
ax2.set_ylabel("Amplitude")
ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
ax2.axhline(Lambda_DE, color='blue', linestyle=':', label='$\Lambda$ (Cosmological Constant)')
ax2.axhline(-Lambda_DE, color='red', linestyle=':', label='$-\Lambda$ (Negative Pressure)')

ax2.legend(loc='center right')
ax2.grid(True, alpha=0.3)

# Annotations for Plot 2
ax2.annotate('Falling phase\n(Inflation)', xy=(2, max(rho_t)), xytext=(5, max(rho_t)-0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
ax2.annotate('Stabilization\n(P = -$\\rho$)', xy=(20, -Lambda_DE), xytext=(15, -Lambda_DE+0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig("dark_energy_proof.png", dpi=300)
plt.show()