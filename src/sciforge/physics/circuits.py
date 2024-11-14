from typing import Optional, List, Dict, Union, Tuple
import numpy as np
from numpy.typing import ArrayLike
from .base import PhysicalSystem
from .fields import ElectricField, MagneticField
from ..core.constants import CONSTANTS

class CircuitElement:
    """Base class for circuit elements"""
    def __init__(self, value: float):
        self.value = value
        self.current = 0.0
        self.voltage = 0.0
        self.power = 0.0
        self.frequency = 0.0
        
    def update_state(self):
        """Update element's power based on current and voltage"""
        self.power = self.voltage * np.abs(self.current)  # Handle complex currents
        
    def impedance(self, frequency: float = 0.0) -> complex:
        """Calculate complex impedance at given frequency"""
        raise NotImplementedError
        
    def admittance(self, frequency: float = 0.0) -> complex:
        """Calculate complex admittance at given frequency"""
        return 1 / self.impedance(frequency)

class VoltageSource(CircuitElement):
    """Ideal voltage source with optional time-varying voltage"""
    def __init__(self, voltage: float, internal_resistance: float = 0.0,
                 amplitude: float = 0.0, frequency: float = 0.0, phase: float = 0.0):
        super().__init__(internal_resistance)
        self.source_voltage = voltage
        self.amplitude = amplitude
        self.frequency = frequency 
        self.phase = phase
        
    def impedance(self, frequency: float = 0.0) -> complex:
        return complex(self.value, 0)  # Internal resistance
        
    def get_voltage(self, current: float, time: float = 0.0) -> complex:
        """Get terminal voltage accounting for internal resistance and AC component"""
        ac_voltage = self.amplitude * np.exp(1j * (2*np.pi*self.frequency*time + self.phase))
        return (self.source_voltage + ac_voltage) - current * self.value

class CurrentSource(CircuitElement):
    """Ideal current source with optional time-varying current"""
    def __init__(self, current: float, internal_conductance: float = 0.0,
                 amplitude: float = 0.0, frequency: float = 0.0, phase: float = 0.0):
        super().__init__(1/internal_conductance if internal_conductance else float('inf'))
        self.source_current = current
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        
    def impedance(self, frequency: float = 0.0) -> complex:
        return complex(self.value, 0)
        
    def get_current(self, time: float = 0.0) -> complex:
        """Get source current including AC component"""
        ac_current = self.amplitude * np.exp(1j * (2*np.pi*self.frequency*time + self.phase))
        return self.source_current + ac_current

class RLCElement(CircuitElement):
    """Combined RLC circuit element with parasitic effects"""
    def __init__(self, resistance: float = 0.0, inductance: float = 0.0, 
                 capacitance: float = 0.0, parasitic_r: float = 0.0,
                 parasitic_l: float = 0.0, parasitic_c: float = 0.0):
        super().__init__(resistance)
        self.inductance = inductance
        self.capacitance = capacitance
        self.parasitic_r = parasitic_r
        self.parasitic_l = parasitic_l
        self.parasitic_c = parasitic_c
        
    def impedance(self, frequency: float = 0.0) -> complex:
        if frequency == 0:
            return complex(self.value + self.parasitic_r, 0)
        
        omega = 2 * np.pi * frequency
        # Main component impedances
        z_r = complex(self.value, 0)
        z_l = 1j * omega * self.inductance if self.inductance else 0
        z_c = 1/(1j * omega * self.capacitance) if self.capacitance else 0
        
        # Parasitic impedances
        z_pr = complex(self.parasitic_r, 0)
        z_pl = 1j * omega * self.parasitic_l if self.parasitic_l else 0
        z_pc = 1/(1j * omega * self.parasitic_c) if self.parasitic_c else 0
        
        return z_r + z_l + z_c + z_pr + z_pl + z_pc

class CircuitNetwork:
    """Network of interconnected circuit elements with AC/DC analysis"""
    def __init__(self):
        self.nodes: List[Dict[str, Union[float, List[CircuitElement]]]] = []
        self.elements: List[CircuitElement] = []
        self.connections: Dict[int, List[Tuple[int, CircuitElement]]] = {}
        self.time = 0.0
        self.frequency = 0.0
        
    def add_node(self, voltage: float = 0.0) -> int:
        """Add node to circuit and return its index"""
        self.nodes.append({
            'voltage': complex(voltage, 0),
            'elements': []
        })
        return len(self.nodes) - 1
        
    def connect(self, node1: int, node2: int, element: CircuitElement):
        """Connect two nodes with a circuit element"""
        if node1 not in self.connections:
            self.connections[node1] = []
        if node2 not in self.connections:
            self.connections[node2] = []
            
        self.connections[node1].append((node2, element))
        self.connections[node2].append((node1, element))
        
        # Add element to nodes
        self.nodes[node1]['elements'].append(element)
        self.nodes[node2]['elements'].append(element)
        
        # Add to elements list if new
        if element not in self.elements:
            self.elements.append(element)
            
    def solve_dc(self) -> Dict[str, np.ndarray]:
        """Solve DC circuit using modified nodal analysis"""
        n = len(self.nodes)
        conductance = np.zeros((n, n), dtype=complex)
        current = np.zeros(n, dtype=complex)
        
        # Build conductance matrix and current vector
        for i in range(n):
            for j, element in self.connections.get(i, []):
                if isinstance(element, VoltageSource):
                    conductance[i,i] += 1/element.value
                    conductance[i,j] -= 1/element.value
                    current[i] += element.get_voltage(0, self.time)/element.value
                elif isinstance(element, CurrentSource):
                    current[i] += element.get_current(self.time)
                else:
                    conductance[i,i] += 1/element.impedance(0)
                    conductance[i,j] -= 1/element.impedance(0)
                    
        # Solve system with LU decomposition for better stability
        voltages = np.zeros(n, dtype=complex)
        if n > 1:
            voltages[1:] = np.linalg.solve(conductance[1:,1:], current[1:])
        
        # Update element states
        self._update_elements(voltages)
        
        return {
            'voltages': voltages,
            'currents': np.array([elem.current for elem in self.elements]),
            'powers': np.array([elem.power for elem in self.elements])
        }