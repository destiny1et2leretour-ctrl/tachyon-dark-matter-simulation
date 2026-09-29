# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:43:46 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt

G = 1.0              # Gravitational constant (simplified)
M_dark = 1000.0      # Mass of dark matter (located in U_i3D)
N_stars = 50         # Number of baryonic stars (in U_3D)
dt = 0.01            # Time step
steps = 2000         # Number of iterations

def initialize_stars():
    """Generates a disk of stars in U_3D with tangential velocity"""
    theta = np.random.uniform(0, 2*np.pi, N_stars)
    r = np.random.uniform(20, 50, N_stars)
    
    # Initial positions (x, y) in the real plane U_3D
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Initial velocities (for a perfect circular orbit if attracted by M_dark)
    # v = sqrt(G * M / r)
    v_mag = np.sqrt(G * M_dark / r)
    vx = -v_mag * np.sin(theta)
    vy = v_mag * np.cos(theta)
    
    return np.column_stack((x, y)), np.column_stack((vx, vy))

def simulate_universe(mu_coupling):
    """
    Simulates the trajectory of the stars.
    mu_coupling = the \mu term of the matrix G(p).
    If mu = 0, the two universes ignore each other.
    If mu > 0, U_i3D attracts U_3D.
    """
    pos, vel = initialize_stars()
    
    # History of positions to plot the orbits
    history_x = np.zeros((steps, N_stars))
    history_y = np.zeros((steps, N_stars))
    
    # Position of the dark mass (0,0) but in the imaginary space U_i3D
    pos_dark = np.array([0.0, 0.0])
    
    for t in range(steps):
        history_x[t, :] = pos[:, 0]
        history_y[t, :] = pos[:, 1]
        
        # Calculation of the force according to the C^3 geometry
        # The acceleration felt by a star in U_3D depends on the mass in U_i3D * mu
        r_vec = pos_dark - pos
        distances = np.linalg.norm(r_vec, axis=1)
        
        # Newton's law modified by the Hermitian Tensor: F = (G * M_dark / r^2) * mu
        acc_mag = (G * M_dark / distances**2) * mu_coupling
        
        # Acceleration vectors
        ax = acc_mag * (r_vec[:, 0] / distances)
        ay = acc_mag * (r_vec[:, 1] / distances)
        
        # Semi-implicit Euler integration (update v then x)
        vel[:, 0] += ax * dt
        vel[:, 1] += ay * dt
        pos[:, 0] += vel[:, 0] * dt
        pos[:, 1] += vel[:, 1] * dt
        
    return history_x, history_y

np.random.seed(42) # To have the same initial positions

# Scenario A: Without the Graviton-Bridge (Classical physics)
hist_x_0, hist_y_0 = simulate_universe(mu_coupling=0.0)

# Scenario B: With the Graviton-Bridge (C^3 model)
hist_x_1, hist_y_1 = simulate_universe(mu_coupling=1.0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='black')
fig.suptitle("N-Body Proof: Tensor-Hermitian Coupling between $U_{3D}$ and $U_{i3D}$", 
             color='white', fontsize=16)

def configure_axis(ax, title):
    ax.set_facecolor('black')
    ax.set_title(title, color='white', pad=20)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.tick_params(colors='white')
    ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

# Plot 1: mu = 0
configure_axis(ax1, "Standard Gravity ($\mu = 0$)\nNo visible mass at the center")
for i in range(N_stars):
    ax1.plot(hist_x_0[:, i], hist_y_0[:, i], color='cyan', alpha=0.3, linewidth=1)
    ax1.scatter(hist_x_0[-1, i], hist_y_0[-1, i], color='white', s=5)
# Empty center
ax1.scatter(0, 0, color='black', edgecolor='red', s=100, marker='x', label="Center $U_{3D}$ (Empty)")
ax1.legend(loc='upper right', facecolor='black', labelcolor='white')

# Plot 2: mu = 1
configure_axis(ax2, "Geometry $\mathbb{C}^3$ with Graviton-Bridge ($\mu = 1$)\nDark Matter in $U_{i3D}$ attracts stars")
for i in range(N_stars):
    ax2.plot(hist_x_1[:, i], hist_y_1[:, i], color='cyan', alpha=0.6, linewidth=1)
    ax2.scatter(hist_x_1[-1, i], hist_y_1[-1, i], color='white', s=5)
# Dark mass (represented in purple because it's in the other universe)
ax2.scatter(0, 0, color='purple', s=200, alpha=0.8, edgecolors='white', 
            label="Dark Matter Halo (in $U_{i3D}$)")
ax2.legend(loc='upper right', facecolor='black', labelcolor='white')

plt.tight_layout()
plt.savefig("graviton_bridge_proof.png", dpi=300, facecolor='black')
plt.show()