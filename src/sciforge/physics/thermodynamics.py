import numpy as np
from typing import Optional, List, Dict
from numpy.typing import ArrayLike

class ThermalSystem:
    """Class representing a thermal system with heat transfer capabilities"""
    
    def __init__(self, 
                 temperature: float,
                 mass: float,
                 specific_heat: float,
                 thermal_conductivity: Optional[float] = None):
        """
        Initialize thermal system
        
        Args:
            temperature: Initial temperature in Kelvin
            mass: Mass of system in kg
            specific_heat: Specific heat capacity in J/(kg·K)
            thermal_conductivity: Thermal conductivity in W/(m·K), optional
        """
        self.temperature = temperature
        self.mass = mass
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.history = {'time': [], 'temperature': []}
        
    def heat_energy(self) -> float:
        """Calculate total heat energy of system"""
        return self.mass * self.specific_heat * self.temperature
    
    def add_heat(self, heat: float):
        """
        Add heat energy to system
        
        Args:
            heat: Heat energy to add in Joules
        """
        delta_t = heat / (self.mass * self.specific_heat)
        self.temperature += delta_t
        
    def conductive_heat_transfer(self, 
                               other: 'ThermalSystem',
                               contact_area: float,
                               distance: float,
                               time: float) -> float:
        """
        Calculate conductive heat transfer with another system
        
        Args:
            other: Other thermal system
            contact_area: Contact area between systems in m²
            distance: Distance between system centers in m
            time: Time duration in seconds
            
        Returns:
            Heat energy transferred in Joules
        """
        if self.thermal_conductivity is None or other.thermal_conductivity is None:
            raise ValueError("Both systems must have thermal conductivity defined")
            
        # Effective thermal conductivity
        k_eff = (self.thermal_conductivity * other.thermal_conductivity) / \
                (self.thermal_conductivity + other.thermal_conductivity)
                
        # Heat transfer rate
        q = k_eff * contact_area * (self.temperature - other.temperature) / distance
        
        # Total heat transferred
        heat = q * time
        
        # Update temperatures
        self.add_heat(-heat)
        other.add_heat(heat)
        
        return heat
    
    def update_history(self, time: float):
        """
        Update temperature history
        
        Args:
            time: Current time
        """
        self.history['time'].append(time)
        self.history['temperature'].append(self.temperature)
