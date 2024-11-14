"""
Example demonstrating attosecond pulse generation through high harmonic generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.attosecond import AttosecondPulseGenerator
from src.sciforge.core.constants import CONSTANTS

def simulate_hhg():
    # Create spatial grid
    x = np.linspace(-50e-9, 50e-9, 200)  # 100 nm range
    
    # Create initial wavefunction (ground state gaussian)
    psi0 = np.exp(-(x/1e-9)**2)  # 1 nm width gaussian
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2))  # Normalize
    
    # Create attosecond pulse generator
    hhg = AttosecondPulseGenerator(
        x=x,
        wavelength=800e-9,    # 800 nm Ti:Sapphire laser
        intensity=1e14,       # 10¹⁴ W/cm²
        pulse_duration=5e-15, # 5 fs pulse
        wavefunction=psi0     # Add initial wavefunction
    )
    
    # Time grid for simulation
    t = np.linspace(-10e-15, 10e-15, 1000)  # ±10 fs
    
    # Generate attosecond pulse
    t_as, attosecond_field = hhg.generate_attosecond_pulse(t, harmonic_range=(11, 31))
    
    return hhg, t_as, attosecond_field

def plot_results(hhg, t_as, attosecond_field):
    """Create comprehensive visualization of the HHG process"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot driving laser field evolution
    t = np.array(hhg.history['time'])
    # Reshape position and time for proper broadcasting
    x_reshaped = hhg.position[:, np.newaxis]  # Shape: (200, 1)
    t_reshaped = t[np.newaxis, :]            # Shape: (1, len(t))
    E_t = hhg.driving_laser.wavefunction(x_reshaped, t_reshaped)
    E_t = hhg.E0 * np.real(E_t[0, :])  # Take first spatial point and scale by peak field
    ax1.plot(t * 1e15, E_t * 1e-11)
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Electric Field (×10¹¹ V/m)')
    ax1.set_title('Driving Laser Field')
    ax1.grid(True)
    
    # Plot dipole moment evolution
    ax2.plot(t[:-1] * 1e15, hhg.history['dipole_moment'])
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Dipole Moment (C⋅m)')
    ax2.set_title('Dipole Moment Evolution')
    ax2.grid(True)
    
    # Plot ionization rate
    ax3.semilogy(t[:-1] * 1e15, hhg.history['ionization_rate'])
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Ionization Rate (s⁻¹)')
    ax3.set_title('Tunnel Ionization Rate')
    ax3.grid(True)
    
    # Plot attosecond pulse
    ax4.plot(t_as * 1e18, np.abs(attosecond_field))
    ax4.set_xlabel('Time (as)')
    ax4.set_ylabel('Field Strength (arb. units)')
    ax4.set_title('Attosecond Pulse')
    ax4.grid(True)
    
    plt.tight_layout()

def main():
    # Run simulation
    hhg, t_as, attosecond_field = simulate_hhg()
    
    # Plot results
    plot_results(hhg, t_as, attosecond_field)
    plt.show()

if __name__ == "__main__":
    main()