import numpy as np
from typing import Optional, List, Dict
from numpy.typing import ArrayLike
from .base import ThermodynamicSystem

class ThermalSystem(ThermodynamicSystem):
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
        super().__init__(mass=mass,
                        position=np.array([0.0]),
                        temperature=temperature,
                        specific_heat=specific_heat)
        self.thermal_conductivity = thermal_conductivity
        self.history = {'time': [], 'temperature': []}
        
    def heat_energy(self) -> float:
        """Calculate total heat energy of system"""
        return self.energy()
    
    def add_heat(self, heat: float):
        """
        Add heat energy to system
        
        Args:
            heat: Heat energy to add in Joules
        """
        super().add_heat(heat)
        
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

class IdealGas(ThermalSystem):
    """Class representing an ideal gas system"""
    
    def __init__(self,
                 temperature: float,
                 pressure: float,
                 volume: float,
                 molar_mass: float,
                 n_moles: float):
        """
        Initialize ideal gas system
        
        Args:
            temperature: Initial temperature in Kelvin
            pressure: Initial pressure in Pascal
            volume: Volume in m³
            molar_mass: Molar mass in kg/mol
            n_moles: Number of moles
        """
        mass = molar_mass * n_moles
        # Using monatomic ideal gas specific heat
        specific_heat = 3 * 8.314 / (2 * molar_mass)  # J/(kg·K)
        super().__init__(temperature, mass, specific_heat)
        self.pressure = pressure
        self.volume = volume
        self.n_moles = n_moles
        self.R = 8.314  # Gas constant J/(mol·K)
        
    def update_state(self):
        """Update pressure using ideal gas law PV = nRT"""
        self.pressure = self.n_moles * self.R * self.temperature / self.volume

class HeatExchanger(ThermalSystem):
    """Class representing a heat exchanger system"""
    
    def __init__(self,
                 temperature: float,
                 mass: float,
                 specific_heat: float,
                 flow_rate: float,
                 efficiency: float = 0.8):
        """
        Initialize heat exchanger
        
        Args:
            temperature: Initial temperature in Kelvin
            mass: Mass of fluid in kg
            specific_heat: Specific heat capacity in J/(kg·K)
            flow_rate: Fluid flow rate in kg/s
            efficiency: Heat exchanger efficiency (0-1)
        """
        super().__init__(temperature, mass, specific_heat)
        self.flow_rate = flow_rate
        self.efficiency = efficiency
        
    def heat_transfer_rate(self, other: 'HeatExchanger') -> float:
        """Calculate heat transfer rate between two fluid streams"""
        c_min = min(self.flow_rate * self.specific_heat,
                   other.flow_rate * other.specific_heat)
        return self.efficiency * c_min * (self.temperature - other.temperature)

class PhaseChangeSystem(ThermalSystem):
    """Class representing a system that can undergo phase changes"""
    
    def __init__(self,
                 temperature: float,
                 mass: float,
                 specific_heat: float,
                 latent_heat: float,
                 melting_point: float):
        """
        Initialize phase change system
        
        Args:
            temperature: Initial temperature in Kelvin
            mass: Mass in kg
            specific_heat: Specific heat capacity in J/(kg·K)
            latent_heat: Latent heat of fusion/vaporization in J/kg
            melting_point: Melting/boiling point in Kelvin
        """
        super().__init__(temperature, mass, specific_heat)
        self.latent_heat = latent_heat
        self.melting_point = melting_point
        self.phase = 'solid' if temperature < melting_point else 'liquid'
        
    def add_heat(self, heat: float):
        """Override to handle phase changes"""
        if heat > 0 and self.temperature == self.melting_point and self.phase == 'solid':
            # Calculate energy needed for complete phase change
            phase_change_energy = self.mass * self.latent_heat
            if heat < phase_change_energy:
                return  # Still in phase change
            heat -= phase_change_energy
            self.phase = 'liquid'
        super().add_heat(heat)
