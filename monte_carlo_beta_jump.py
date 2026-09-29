# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:45:24 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N_particles = 500000       # Number of simulated dark matter particles
R_galaxy = 15.0            # Galaxy radius (in kpc)
alpha_friction = 0.8       # Dark energy coupling constant (lambda * phi)

def generate_dark_matter_halo(N):
    """
    Generates the dark matter distribution (Simplified NFW profile).
    Dark matter is highly dense at the center and diffuse at the edges.
    """
    # Radial distribution concentrated at the center (exponential/normal law)
    r = np.abs(np.random.normal(0, R_galaxy/3, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, r

def beta_jump_probability(r):
    """
    Calculates the probability that a particle undergoes a Beta Jump.
    According to the model, Gamma depends on friction (local energy density).
    The closer to the center (small r), the stronger the friction.
    """
    # Probability diverges at the center (r -> 0) but is capped by the core (epsilon)
    epsilon = 0.5 
    prob = alpha_friction / (r**2 + epsilon)
    
    # Ensure probability remains between 0 and 1 (it's a transition rate)
    return np.clip(prob * 0.01, 0, 1)

np.random.seed(42)

# Step A: Place dark matter in the complex universe U_i3D (Invisible)
x_dm, y_dm, r_dm = generate_dark_matter_halo(N_particles)

# Step B: Calculate the jump probability for EACH particle
probabilities = beta_jump_probability(r_dm)

# Step C: Quantum dice roll (Monte-Carlo)
# If the random number is less than the probability, the particle makes a Beta Jump
quantum_draw = np.random.random(N_particles)
successful_jumps = quantum_draw < probabilities

# Isolate the coordinates of the particles that emitted a gamma flash
x_flash = x_dm[successful_jumps]
y_flash = y_dm[successful_jumps]

print(f"Out of {N_particles} dark particles, only {len(x_flash)} made a Beta Jump.")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
fig.suptitle("Beta Jump Simulation $\\rightarrow$ Galactic Gamma Emission", 
             color='white', fontsize=16)

def configure_axis(ax, title):
    ax.set_facecolor('black')
    ax.set_title(title, color='white', pad=15)
    ax.set_xlim(-R_galaxy, R_galaxy)
    ax.set_ylim(-R_galaxy, R_galaxy)
    ax.tick_params(colors='white')
    ax.set_xlabel("Distance (kpc)", color='white')
    ax.set_ylabel("Distance (kpc)", color='white')

# ---- Plot 1: Theoretical Reality (Dark Matter in U_i3D) ----
configure_axis(ax1, "Dark Matter Distribution ($U_{i3D}$)\n(Invisible to our telescopes)")
# Display only a subsample to avoid saturating the image
ax1.scatter(x_dm[::10], y_dm[::10], color='purple', alpha=0.05, s=1)
ax1.text(-14, 13, f"Simulated particles: {N_particles}", color='purple', fontsize=10)

# ---- Plot 2: Telescopic Observation (Flashes in U_3D) ----
configure_axis(ax2, "Gamma Flash Map ($U_{3D}$)\n(What the Fermi-LAT telescope observes)")
# Use a 2D histogram to create a heatmap
h = ax2.hist2d(x_flash, y_flash, bins=100, cmap='magma', norm=LogNorm())
plt.colorbar(h[3], ax=ax2, label="Gamma Ray Intensity (Counts/pixel)")

# Annotation for the galactic excess
ax2.annotate('Gamma Excess at Center\n(Maximum quantum friction)', 
             xy=(0, 0), xytext=(3, 4), color='white',
             arrowprops=dict(facecolor='white', arrowstyle='->', color='white'))

plt.tight_layout()
plt.savefig("beta_jump_fermi_proof.png", dpi=300, facecolor='black')
plt.show()