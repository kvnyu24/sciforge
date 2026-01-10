"""
Thermodynamics & Statistical Mechanics Module

This module implements comprehensive thermodynamics and statistical mechanics:
- Thermodynamic Laws and Processes
- Thermodynamic Potentials
- Equations of State
- Statistical Ensembles
- Quantum Statistics
- Phase Transitions
- Non-equilibrium Thermodynamics

References:
    - Callen, "Thermodynamics and an Introduction to Thermostatistics"
    - Pathria & Beale, "Statistical Mechanics"
    - Landau & Lifshitz, "Statistical Physics"
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from dataclasses import dataclass
from numpy.typing import ArrayLike

from ..core.base import BaseClass, BaseSolver
from ..core.utils import validate_positive
from ..core.exceptions import ValidationError, PhysicsError


# ==============================================================================
# Physical Constants
# ==============================================================================

K_B = 1.380649e-23      # Boltzmann constant (J/K)
N_A = 6.02214076e23     # Avogadro's number
R = 8.314462            # Gas constant (J/(mol·K))
H = 6.62607015e-34      # Planck constant (J·s)
HBAR = H / (2 * np.pi)  # Reduced Planck constant


# ==============================================================================
# Phase 4.1: Thermodynamic Laws
# ==============================================================================

class ThermodynamicProcess(BaseClass):
    """
    General thermodynamic process between states.

    Supports isothermal, adiabatic, isobaric, and isochoric processes.

    Args:
        n_moles: Number of moles of gas
        gamma: Heat capacity ratio Cp/Cv (default 1.4 for diatomic)
        process_type: 'isothermal', 'adiabatic', 'isobaric', 'isochoric'

    Examples:
        >>> process = ThermodynamicProcess(n_moles=1.0, process_type='isothermal')
        >>> W = process.work(V1=0.001, V2=0.002, T=300)
    """

    def __init__(
        self,
        n_moles: float = 1.0,
        gamma: float = 1.4,
        process_type: str = 'isothermal'
    ):
        super().__init__()

        validate_positive(n_moles, "n_moles")
        validate_positive(gamma, "gamma")

        self.n = n_moles
        self.gamma = gamma
        self.process_type = process_type

        # Heat capacities (per mole)
        self.Cv = R / (gamma - 1)
        self.Cp = gamma * self.Cv

    def work(
        self,
        V1: float,
        V2: float,
        T: Optional[float] = None,
        P: Optional[float] = None
    ) -> float:
        """
        Calculate work done during process.

        W = ∫ P dV

        Args:
            V1, V2: Initial and final volumes (m³)
            T: Temperature (K) for isothermal
            P: Pressure (Pa) for isobaric
        """
        if self.process_type == 'isothermal':
            if T is None:
                raise ValidationError("Temperature required for isothermal process")
            # W = nRT ln(V2/V1)
            return self.n * R * T * np.log(V2 / V1)

        elif self.process_type == 'adiabatic':
            if P is None and T is not None:
                P = self.n * R * T / V1
            elif P is None:
                raise ValidationError("Need P or T for adiabatic process")
            # W = (P1V1 - P2V2) / (γ - 1)
            P2 = P * (V1 / V2)**self.gamma
            return (P * V1 - P2 * V2) / (self.gamma - 1)

        elif self.process_type == 'isobaric':
            if P is None:
                raise ValidationError("Pressure required for isobaric process")
            return P * (V2 - V1)

        elif self.process_type == 'isochoric':
            return 0.0  # No volume change

        else:
            raise ValidationError(f"Unknown process type: {self.process_type}")

    def heat(
        self,
        V1: float,
        V2: float,
        T1: Optional[float] = None,
        T2: Optional[float] = None,
        P: Optional[float] = None
    ) -> float:
        """
        Calculate heat transferred during process.

        Q = ΔU + W (First Law)
        """
        if self.process_type == 'isothermal':
            # Q = W for ideal gas (ΔU = 0)
            return self.work(V1, V2, T=T1, P=P)

        elif self.process_type == 'adiabatic':
            return 0.0  # By definition

        elif self.process_type == 'isobaric':
            if T1 is None or T2 is None:
                raise ValidationError("T1 and T2 required for isobaric heat")
            return self.n * self.Cp * (T2 - T1)

        elif self.process_type == 'isochoric':
            if T1 is None or T2 is None:
                raise ValidationError("T1 and T2 required for isochoric heat")
            return self.n * self.Cv * (T2 - T1)

        else:
            raise ValidationError(f"Unknown process type: {self.process_type}")

    def entropy_change(
        self,
        V1: float,
        V2: float,
        T1: float,
        T2: Optional[float] = None
    ) -> float:
        """
        Calculate entropy change during process.

        ΔS = ∫ dQ/T
        """
        if self.process_type == 'isothermal':
            # ΔS = nR ln(V2/V1)
            return self.n * R * np.log(V2 / V1)

        elif self.process_type == 'adiabatic':
            return 0.0  # Reversible adiabatic is isentropic

        elif self.process_type == 'isobaric':
            if T2 is None:
                raise ValidationError("T2 required for isobaric entropy")
            return self.n * self.Cp * np.log(T2 / T1)

        elif self.process_type == 'isochoric':
            if T2 is None:
                raise ValidationError("T2 required for isochoric entropy")
            return self.n * self.Cv * np.log(T2 / T1)

        else:
            raise ValidationError(f"Unknown process type: {self.process_type}")


class CarnotEngine(BaseClass):
    """
    Ideal Carnot heat engine.

    Operates between hot reservoir (T_H) and cold reservoir (T_C).

    Args:
        T_hot: Hot reservoir temperature (K)
        T_cold: Cold reservoir temperature (K)
        n_moles: Moles of working substance

    Examples:
        >>> engine = CarnotEngine(T_hot=500, T_cold=300)
        >>> eta = engine.efficiency()  # 0.4
        >>> W = engine.work_per_cycle(Q_H=1000)
    """

    def __init__(
        self,
        T_hot: float,
        T_cold: float,
        n_moles: float = 1.0
    ):
        super().__init__()

        validate_positive(T_hot, "T_hot")
        validate_positive(T_cold, "T_cold")

        if T_cold >= T_hot:
            raise PhysicsError("T_cold must be less than T_hot")

        self.T_H = T_hot
        self.T_C = T_cold
        self.n = n_moles

    def efficiency(self) -> float:
        """
        Calculate Carnot efficiency.

        η = 1 - T_C/T_H
        """
        return 1 - self.T_C / self.T_H

    def work_per_cycle(self, Q_H: float) -> float:
        """Calculate work output per cycle given heat input Q_H."""
        return self.efficiency() * Q_H

    def heat_rejected(self, Q_H: float) -> float:
        """Calculate heat rejected to cold reservoir."""
        return Q_H - self.work_per_cycle(Q_H)

    def coefficient_of_performance_refrigerator(self) -> float:
        """COP for Carnot refrigerator (reversed cycle)."""
        return self.T_C / (self.T_H - self.T_C)

    def coefficient_of_performance_heat_pump(self) -> float:
        """COP for Carnot heat pump."""
        return self.T_H / (self.T_H - self.T_C)

    def pv_diagram(
        self,
        V1: float,
        V2: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate P-V diagram for Carnot cycle.

        Returns:
            Arrays of (P, V) points around the cycle
        """
        # State 1: Start of isothermal expansion at T_H
        P1 = self.n * R * self.T_H / V1

        # State 2: End of isothermal expansion, start adiabatic
        P2 = self.n * R * self.T_H / V2

        # State 3: End of adiabatic expansion at T_C
        gamma = 1.4
        V3 = V2 * (self.T_H / self.T_C)**(1/(gamma-1))
        P3 = self.n * R * self.T_C / V3

        # State 4: End of isothermal compression
        V4 = V1 * (self.T_H / self.T_C)**(1/(gamma-1))
        P4 = self.n * R * self.T_C / V4

        # Generate points for each process
        V_list = []
        P_list = []

        # Process 1-2: Isothermal expansion at T_H
        V_12 = np.linspace(V1, V2, n_points//4)
        P_12 = self.n * R * self.T_H / V_12
        V_list.extend(V_12)
        P_list.extend(P_12)

        # Process 2-3: Adiabatic expansion
        V_23 = np.linspace(V2, V3, n_points//4)
        P_23 = P2 * (V2 / V_23)**gamma
        V_list.extend(V_23)
        P_list.extend(P_23)

        # Process 3-4: Isothermal compression at T_C
        V_34 = np.linspace(V3, V4, n_points//4)
        P_34 = self.n * R * self.T_C / V_34
        V_list.extend(V_34)
        P_list.extend(P_34)

        # Process 4-1: Adiabatic compression
        V_41 = np.linspace(V4, V1, n_points//4)
        P_41 = P4 * (V4 / V_41)**gamma
        V_list.extend(V_41)
        P_list.extend(P_41)

        return np.array(V_list), np.array(P_list)


class HeatPump(BaseClass):
    """
    Heat pump / refrigeration cycle.

    Args:
        T_hot: Hot side temperature (K)
        T_cold: Cold side temperature (K)
        cycle_type: 'carnot', 'vapor_compression'
    """

    def __init__(
        self,
        T_hot: float,
        T_cold: float,
        cycle_type: str = 'carnot'
    ):
        super().__init__()

        self.T_H = T_hot
        self.T_C = T_cold
        self.cycle_type = cycle_type

    def cop_cooling(self) -> float:
        """
        Coefficient of Performance for cooling (refrigerator mode).

        COP = Q_C / W
        """
        if self.cycle_type == 'carnot':
            return self.T_C / (self.T_H - self.T_C)
        else:
            # Realistic efficiency ~60% of Carnot
            return 0.6 * self.T_C / (self.T_H - self.T_C)

    def cop_heating(self) -> float:
        """
        Coefficient of Performance for heating (heat pump mode).

        COP = Q_H / W = 1 + COP_cooling
        """
        return 1 + self.cop_cooling()

    def power_required(self, heat_rate: float, mode: str = 'cooling') -> float:
        """
        Calculate power required for given heat transfer rate.

        Args:
            heat_rate: Heat transfer rate (W)
            mode: 'cooling' or 'heating'
        """
        if mode == 'cooling':
            return heat_rate / self.cop_cooling()
        else:
            return heat_rate / self.cop_heating()


class EntropyCalculator(BaseClass):
    """
    Entropy calculations for various systems.

    Args:
        system_type: 'ideal_gas', 'solid', 'mixing'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def ideal_gas_entropy(
        self,
        n: float,
        T: float,
        V: float,
        T_ref: float = 298.15,
        V_ref: float = 0.0224,
        Cv: float = 3/2 * R
    ) -> float:
        """
        Calculate entropy of ideal gas relative to reference state.

        S = S_ref + nCv ln(T/T_ref) + nR ln(V/V_ref)
        """
        return n * (Cv * np.log(T / T_ref) + R * np.log(V / V_ref))

    def mixing_entropy(self, n_species: List[float]) -> float:
        """
        Calculate entropy of mixing for ideal mixture.

        ΔS_mix = -R Σ n_i ln(x_i)
        """
        total = sum(n_species)
        entropy = 0.0
        for n in n_species:
            if n > 0:
                x = n / total
                entropy -= n * R * np.log(x)
        return entropy

    def solid_entropy(
        self,
        n: float,
        T: float,
        theta_D: float
    ) -> float:
        """
        Calculate entropy of solid using Debye model.

        S = 3nR [4D(θ_D/T) - 3 ln(1 - e^(-θ_D/T))]

        where D is the Debye function.
        """
        x = theta_D / T
        return 3 * n * R * (4 * self._debye_function(x) - 3 * np.log(1 - np.exp(-x) + 1e-30))

    def _debye_function(self, x: float, n_terms: int = 100) -> float:
        """Compute Debye function D_3(x)."""
        if x < 0.01:
            return 1 - 3*x/8 + x**2/20
        if x > 20:
            return 0.0

        # Numerical integration
        t = np.linspace(0.001, x, n_terms)
        dt = t[1] - t[0]
        integrand = t**3 / (np.exp(t) - 1 + 1e-30)
        return 3 * np.sum(integrand) * dt / x**3


class FreeEnergyMinimizer(BaseClass):
    """
    Minimization of thermodynamic free energies.

    Finds equilibrium states by minimizing Helmholtz or Gibbs free energy.

    Args:
        free_energy_func: Function F(x) or G(x) to minimize
        constraints: List of constraint functions
    """

    def __init__(
        self,
        free_energy_func: Callable[[np.ndarray], float],
        constraints: Optional[List[Callable]] = None
    ):
        super().__init__()

        self.F = free_energy_func
        self.constraints = constraints or []

    def minimize(
        self,
        x0: ArrayLike,
        method: str = 'gradient_descent',
        tol: float = 1e-8,
        max_iter: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Find free energy minimum.

        Args:
            x0: Initial guess
            method: 'gradient_descent' or 'newton'
            tol: Convergence tolerance
            max_iter: Maximum iterations
            learning_rate: Step size for gradient descent

        Returns:
            (x_min, F_min)
        """
        x = np.array(x0, dtype=float)

        for iteration in range(max_iter):
            # Numerical gradient
            grad = self._gradient(x)

            if method == 'gradient_descent':
                x_new = x - learning_rate * grad
            else:
                # Newton's method with numerical Hessian
                H = self._hessian(x)
                try:
                    step = np.linalg.solve(H, grad)
                    x_new = x - step
                except np.linalg.LinAlgError:
                    x_new = x - learning_rate * grad

            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                return x_new, self.F(x_new)

            x = x_new

        return x, self.F(x)

    def _gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Numerical gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self.F(x_plus) - self.F(x_minus)) / (2 * eps)
        return grad

    def _hessian(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerical Hessian."""
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps

                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps

                H[i, j] = (self.F(x_pp) - self.F(x_pm) - self.F(x_mp) + self.F(x_mm)) / (4 * eps**2)
        return H


# ==============================================================================
# Phase 4.2: Thermodynamic Potentials
# ==============================================================================

class InternalEnergy(BaseClass):
    """
    Internal energy U(S, V, N) calculations.

    Args:
        system_type: 'ideal_gas', 'van_der_waals', 'einstein_solid'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def U_ideal_gas(self, n: float, T: float, dof: int = 3) -> float:
        """Internal energy of ideal gas: U = (f/2) n R T."""
        return (dof / 2) * n * R * T

    def U_einstein_solid(self, N: int, T: float, omega_E: float) -> float:
        """
        Internal energy of Einstein solid.

        U = 3N ℏω_E [1/2 + 1/(e^(ℏω_E/kT) - 1)]
        """
        x = HBAR * omega_E / (K_B * T)
        return 3 * N * HBAR * omega_E * (0.5 + 1 / (np.exp(x) - 1 + 1e-30))

    def U_van_der_waals(
        self,
        n: float,
        T: float,
        V: float,
        a: float,
        dof: int = 3
    ) -> float:
        """
        Internal energy of van der Waals gas.

        U = (f/2) nRT - a n²/V
        """
        return (dof / 2) * n * R * T - a * n**2 / V


class Enthalpy(BaseClass):
    """
    Enthalpy H = U + PV calculations.

    Args:
        system_type: 'ideal_gas', 'real_gas'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def H_ideal_gas(self, n: float, T: float, dof: int = 3) -> float:
        """Enthalpy of ideal gas: H = ((f+2)/2) n R T."""
        return ((dof + 2) / 2) * n * R * T

    def enthalpy_change(
        self,
        n: float,
        T1: float,
        T2: float,
        Cp: float
    ) -> float:
        """Calculate enthalpy change: ΔH = ∫ Cp dT."""
        return n * Cp * (T2 - T1)

    def enthalpy_of_reaction(
        self,
        H_products: List[float],
        H_reactants: List[float]
    ) -> float:
        """Calculate enthalpy of reaction: ΔH = Σ H_products - Σ H_reactants."""
        return sum(H_products) - sum(H_reactants)


class HelmholtzFree(BaseClass):
    """
    Helmholtz free energy F = U - TS.

    Natural variables: T, V, N

    Args:
        system_type: 'ideal_gas', 'quantum_oscillator'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def F_ideal_gas(
        self,
        n: float,
        T: float,
        V: float,
        m: float = 4.65e-26  # N2 molecule mass
    ) -> float:
        """
        Helmholtz free energy of ideal gas.

        F = -nRT [ln(V/nΛ³) + 1]

        where Λ = h/√(2πmkT) is thermal de Broglie wavelength.
        """
        Lambda = H / np.sqrt(2 * np.pi * m * K_B * T)
        return -n * R * T * (np.log(V / (n * N_A * Lambda**3)) + 1)

    def F_quantum_oscillator(self, N: int, T: float, omega: float) -> float:
        """
        Helmholtz free energy of quantum harmonic oscillators.

        F = N [ℏω/2 + kT ln(1 - e^(-ℏω/kT))]
        """
        x = HBAR * omega / (K_B * T)
        return N * K_B * T * (x/2 + np.log(1 - np.exp(-x) + 1e-30))

    def pressure_from_F(
        self,
        F_func: Callable[[float], float],
        V: float,
        dV: float = 1e-8
    ) -> float:
        """Calculate pressure: P = -(∂F/∂V)_T."""
        return -(F_func(V + dV) - F_func(V - dV)) / (2 * dV)

    def entropy_from_F(
        self,
        F_func: Callable[[float], float],
        T: float,
        dT: float = 1e-6
    ) -> float:
        """Calculate entropy: S = -(∂F/∂T)_V."""
        return -(F_func(T + dT) - F_func(T - dT)) / (2 * dT)


class GibbsFree(BaseClass):
    """
    Gibbs free energy G = H - TS = U + PV - TS.

    Natural variables: T, P, N

    Args:
        system_type: 'ideal_gas', 'mixture'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def G_ideal_gas(
        self,
        n: float,
        T: float,
        P: float,
        P_ref: float = 1e5  # Standard pressure
    ) -> float:
        """
        Gibbs free energy of ideal gas relative to reference.

        G = G° + nRT ln(P/P°)
        """
        return n * R * T * np.log(P / P_ref)

    def G_mixing(
        self,
        n_species: List[float],
        T: float
    ) -> float:
        """
        Gibbs free energy of mixing.

        ΔG_mix = RT Σ n_i ln(x_i)
        """
        total = sum(n_species)
        G_mix = 0.0
        for n in n_species:
            if n > 0:
                x = n / total
                G_mix += n * R * T * np.log(x)
        return G_mix

    def equilibrium_constant(self, delta_G: float, T: float) -> float:
        """Calculate equilibrium constant: K = exp(-ΔG°/(RT))."""
        return np.exp(-delta_G / (R * T))


class ChemicalPotential(BaseClass):
    """
    Chemical potential μ = (∂G/∂N)_{T,P}.

    Args:
        system_type: 'ideal_gas', 'ideal_solution'
    """

    def __init__(self, system_type: str = 'ideal_gas'):
        super().__init__()
        self.system_type = system_type

    def mu_ideal_gas(
        self,
        T: float,
        P: float,
        mu_0: float = 0.0,
        P_ref: float = 1e5
    ) -> float:
        """
        Chemical potential of ideal gas.

        μ = μ° + RT ln(P/P°)
        """
        return mu_0 + R * T * np.log(P / P_ref)

    def mu_ideal_solution(
        self,
        T: float,
        x: float,
        mu_0: float = 0.0
    ) -> float:
        """
        Chemical potential in ideal solution.

        μ_i = μ°_i + RT ln(x_i)
        """
        if x <= 0:
            return -np.inf
        return mu_0 + R * T * np.log(x)

    def equilibrium_condition(
        self,
        mu_products: List[float],
        mu_reactants: List[float]
    ) -> float:
        """
        Check chemical equilibrium: Σ μ_products = Σ μ_reactants

        Returns difference (should be ~0 at equilibrium).
        """
        return sum(mu_products) - sum(mu_reactants)


class MaxwellRelations(BaseClass):
    """
    Maxwell thermodynamic relations (cross-derivative identities).

    Derives thermodynamic identities from exact differentials.
    """

    def __init__(self):
        super().__init__()

    def verify_relation_1(
        self,
        entropy_func: Callable[[float, float], float],  # S(T, V)
        pressure_func: Callable[[float, float], float],  # P(T, V)
        T: float,
        V: float,
        eps: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Verify Maxwell relation: (∂S/∂V)_T = (∂P/∂T)_V

        Returns both sides for comparison.
        """
        # (∂S/∂V)_T
        dS_dV = (entropy_func(T, V + eps) - entropy_func(T, V - eps)) / (2 * eps)

        # (∂P/∂T)_V
        dP_dT = (pressure_func(T + eps, V) - pressure_func(T - eps, V)) / (2 * eps)

        return dS_dV, dP_dT

    def verify_relation_2(
        self,
        entropy_func: Callable[[float, float], float],  # S(T, P)
        volume_func: Callable[[float, float], float],   # V(T, P)
        T: float,
        P: float,
        eps: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Verify Maxwell relation: (∂S/∂P)_T = -(∂V/∂T)_P

        Returns both sides for comparison.
        """
        # (∂S/∂P)_T
        dS_dP = (entropy_func(T, P + eps) - entropy_func(T, P - eps)) / (2 * eps)

        # -(∂V/∂T)_P
        neg_dV_dT = -(volume_func(T + eps, P) - volume_func(T - eps, P)) / (2 * eps)

        return dS_dP, neg_dV_dT


# ==============================================================================
# Phase 4.3: Equations of State
# ==============================================================================

class VanDerWaalsGas(BaseClass):
    """
    Van der Waals equation of state for real gases.

    (P + a n²/V²)(V - nb) = nRT

    Args:
        a: Attraction parameter (Pa·m⁶/mol²)
        b: Volume exclusion parameter (m³/mol)
        n_moles: Number of moles

    Examples:
        >>> # CO2: a = 0.364, b = 4.27e-5
        >>> gas = VanDerWaalsGas(a=0.364, b=4.27e-5, n_moles=1)
        >>> P = gas.pressure(V=0.001, T=300)
    """

    # Critical constants for common gases (a in Pa·m⁶/mol², b in m³/mol)
    GASES = {
        'H2': {'a': 0.0245, 'b': 2.65e-5},
        'He': {'a': 0.00346, 'b': 2.37e-5},
        'N2': {'a': 0.137, 'b': 3.87e-5},
        'O2': {'a': 0.138, 'b': 3.18e-5},
        'CO2': {'a': 0.364, 'b': 4.27e-5},
        'H2O': {'a': 0.554, 'b': 3.05e-5},
    }

    def __init__(
        self,
        a: float,
        b: float,
        n_moles: float = 1.0
    ):
        super().__init__()

        self.a = a
        self.b = b
        self.n = n_moles

        # Critical point
        self.T_c = 8 * a / (27 * R * b)
        self.P_c = a / (27 * b**2)
        self.V_c = 3 * n_moles * b

    @classmethod
    def from_gas(cls, gas_name: str, n_moles: float = 1.0) -> 'VanDerWaalsGas':
        """Create from standard gas parameters."""
        if gas_name not in cls.GASES:
            raise ValidationError(f"Unknown gas: {gas_name}")
        params = cls.GASES[gas_name]
        return cls(params['a'], params['b'], n_moles)

    def pressure(self, V: float, T: float) -> float:
        """Calculate pressure from V and T."""
        n = self.n
        return n * R * T / (V - n * self.b) - self.a * n**2 / V**2

    def volume(self, P: float, T: float, V_guess: float = 0.001) -> float:
        """Solve for volume given P and T (Newton's method)."""
        V = V_guess
        for _ in range(100):
            f = self.pressure(V, T) - P
            df = -n * R * T / (V - self.n * self.b)**2 + 2 * self.a * self.n**2 / V**3
            V_new = V - f / df
            if abs(V_new - V) < 1e-12:
                return V_new
            V = V_new
        return V

    def compressibility_factor(self, V: float, T: float) -> float:
        """Calculate compressibility factor Z = PV/(nRT)."""
        P = self.pressure(V, T)
        return P * V / (self.n * R * T)

    def reduced_state(self, P: float, V: float, T: float) -> Tuple[float, float, float]:
        """Calculate reduced coordinates P_r, V_r, T_r."""
        return P / self.P_c, V / self.V_c, T / self.T_c


class RedlichKwong(BaseClass):
    """
    Redlich-Kwong equation of state (improved real gas model).

    P = RT/(V_m - b) - a/(T^0.5 V_m (V_m + b))

    Args:
        a: Attraction parameter
        b: Repulsion parameter
        n_moles: Number of moles
    """

    def __init__(
        self,
        a: float,
        b: float,
        n_moles: float = 1.0
    ):
        super().__init__()

        self.a = a
        self.b = b
        self.n = n_moles

    @classmethod
    def from_critical(cls, T_c: float, P_c: float, n_moles: float = 1.0) -> 'RedlichKwong':
        """Create from critical temperature and pressure."""
        a = 0.42748 * R**2 * T_c**2.5 / P_c
        b = 0.08664 * R * T_c / P_c
        return cls(a, b, n_moles)

    def pressure(self, V: float, T: float) -> float:
        """Calculate pressure."""
        V_m = V / self.n  # Molar volume
        return (R * T / (V_m - self.b) -
                self.a / (np.sqrt(T) * V_m * (V_m + self.b)))


class VirialExpansion(BaseClass):
    """
    Virial equation of state (power series in density).

    PV/(nRT) = 1 + B(T)/V_m + C(T)/V_m² + ...

    Args:
        B: Second virial coefficient (m³/mol) or function B(T)
        C: Third virial coefficient (m⁶/mol²) or function C(T)
        n_moles: Number of moles
    """

    def __init__(
        self,
        B: Union[float, Callable[[float], float]],
        C: Union[float, Callable[[float], float]] = 0.0,
        n_moles: float = 1.0
    ):
        super().__init__()

        self.B_func = (lambda T: B) if isinstance(B, (int, float)) else B
        self.C_func = (lambda T: C) if isinstance(C, (int, float)) else C
        self.n = n_moles

    def B(self, T: float) -> float:
        """Return second virial coefficient at temperature T."""
        return self.B_func(T)

    def C(self, T: float) -> float:
        """Return third virial coefficient at temperature T."""
        return self.C_func(T)

    def compressibility_factor(self, V_m: float, T: float) -> float:
        """Calculate Z = PV/(nRT)."""
        B = self.B(T)
        C = self.C(T)
        return 1 + B / V_m + C / V_m**2

    def pressure(self, V: float, T: float) -> float:
        """Calculate pressure from virial expansion."""
        V_m = V / self.n
        Z = self.compressibility_factor(V_m, T)
        return Z * self.n * R * T / V


class IdealMixture(BaseClass):
    """
    Ideal gas mixture (Dalton's law) and ideal solution (Raoult's law).

    Args:
        components: List of (n_moles, properties) for each component
    """

    def __init__(
        self,
        n_moles_list: List[float],
        molar_masses: Optional[List[float]] = None
    ):
        super().__init__()

        self.n_list = n_moles_list
        self.n_total = sum(n_moles_list)
        self.x_list = [n / self.n_total for n in n_moles_list]  # Mole fractions
        self.M_list = molar_masses or [0.029] * len(n_moles_list)  # Default: air

    def partial_pressure(self, P_total: float, component: int) -> float:
        """Calculate partial pressure (Dalton's law): P_i = x_i P."""
        return self.x_list[component] * P_total

    def vapor_pressure(
        self,
        P_pure: List[float],
        component: int
    ) -> float:
        """Calculate partial vapor pressure (Raoult's law): P_i = x_i P_i°."""
        return self.x_list[component] * P_pure[component]

    def total_vapor_pressure(self, P_pure: List[float]) -> float:
        """Calculate total vapor pressure of mixture."""
        return sum(x * P for x, P in zip(self.x_list, P_pure))

    def average_molar_mass(self) -> float:
        """Calculate mole-fraction weighted average molar mass."""
        return sum(x * M for x, M in zip(self.x_list, self.M_list))

    def entropy_of_mixing(self) -> float:
        """Calculate entropy of mixing: ΔS = -R Σ n_i ln(x_i)."""
        S_mix = 0.0
        for n, x in zip(self.n_list, self.x_list):
            if x > 0:
                S_mix -= n * R * np.log(x)
        return S_mix


# ==============================================================================
# Phase 4.4: Statistical Ensembles
# ==============================================================================

class MicrocanonicalEnsemble(BaseClass):
    """
    Microcanonical ensemble (fixed E, V, N).

    Calculates thermodynamic quantities from density of states Ω(E).

    Args:
        omega_func: Density of states Ω(E)
        E: Total energy
        V: Volume
        N: Number of particles
    """

    def __init__(
        self,
        omega_func: Callable[[float], float],
        E: float,
        V: float,
        N: int
    ):
        super().__init__()

        self.Omega = omega_func
        self.E = E
        self.V = V
        self.N = N

    def entropy(self) -> float:
        """Calculate entropy: S = k_B ln(Ω)."""
        return K_B * np.log(self.Omega(self.E))

    def temperature(self, dE: float = 1e-20) -> float:
        """
        Calculate temperature from 1/T = ∂S/∂E.
        """
        S_plus = K_B * np.log(self.Omega(self.E + dE))
        S_minus = K_B * np.log(self.Omega(self.E - dE))
        dS_dE = (S_plus - S_minus) / (2 * dE)
        return 1 / dS_dE if dS_dE > 0 else np.inf

    def pressure(self, dV: float = 1e-30) -> float:
        """
        Calculate pressure from P/T = ∂S/∂V (requires V-dependent Ω).
        """
        # Would need Omega(E, V) for this
        return 0.0  # Placeholder


class CanonicalEnsemble(BaseClass):
    """
    Canonical ensemble (fixed T, V, N).

    Calculates thermodynamic quantities from partition function Z(T).

    Args:
        energy_levels: Array of energy levels E_i
        degeneracies: Array of degeneracies g_i (default: 1)
        T: Temperature (K)
    """

    def __init__(
        self,
        energy_levels: ArrayLike,
        degeneracies: Optional[ArrayLike] = None,
        T: float = 300.0
    ):
        super().__init__()

        self.E = np.array(energy_levels)
        self.g = np.ones_like(self.E) if degeneracies is None else np.array(degeneracies)
        self.T = T

    def partition_function(self, T: Optional[float] = None) -> float:
        """Calculate canonical partition function Z = Σ g_i exp(-E_i/kT)."""
        T = T or self.T
        beta = 1 / (K_B * T)
        return np.sum(self.g * np.exp(-beta * self.E))

    def probability(self, level: int, T: Optional[float] = None) -> float:
        """Calculate probability of state i: P_i = g_i exp(-E_i/kT) / Z."""
        T = T or self.T
        beta = 1 / (K_B * T)
        Z = self.partition_function(T)
        return self.g[level] * np.exp(-beta * self.E[level]) / Z

    def average_energy(self, T: Optional[float] = None) -> float:
        """Calculate average energy: <E> = Σ E_i P_i."""
        T = T or self.T
        probs = np.array([self.probability(i, T) for i in range(len(self.E))])
        return np.sum(self.E * probs)

    def helmholtz_free_energy(self, T: Optional[float] = None) -> float:
        """Calculate Helmholtz free energy: F = -kT ln(Z)."""
        T = T or self.T
        return -K_B * T * np.log(self.partition_function(T))

    def entropy(self, T: Optional[float] = None) -> float:
        """Calculate entropy: S = -∂F/∂T = k ln(Z) + <E>/T."""
        T = T or self.T
        Z = self.partition_function(T)
        E_avg = self.average_energy(T)
        return K_B * np.log(Z) + E_avg / T

    def heat_capacity(self, T: Optional[float] = None, dT: float = 0.1) -> float:
        """Calculate heat capacity: C_V = ∂<E>/∂T."""
        T = T or self.T
        E_plus = self.average_energy(T + dT)
        E_minus = self.average_energy(T - dT)
        return (E_plus - E_minus) / (2 * dT)


class GrandCanonicalEnsemble(BaseClass):
    """
    Grand canonical ensemble (fixed T, V, μ).

    Allows particle number fluctuations.

    Args:
        energy_levels: Array of energy levels
        particle_numbers: Array of particle numbers for each level
        T: Temperature (K)
        mu: Chemical potential (J)
    """

    def __init__(
        self,
        energy_levels: ArrayLike,
        particle_numbers: ArrayLike,
        T: float = 300.0,
        mu: float = 0.0
    ):
        super().__init__()

        self.E = np.array(energy_levels)
        self.N = np.array(particle_numbers)
        self.T = T
        self.mu = mu

    def grand_partition_function(self, T: Optional[float] = None, mu: Optional[float] = None) -> float:
        """Calculate grand partition function Ξ = Σ exp(-(E - μN)/kT)."""
        T = T or self.T
        mu = mu or self.mu
        beta = 1 / (K_B * T)
        return np.sum(np.exp(-beta * (self.E - mu * self.N)))

    def average_particle_number(self, T: Optional[float] = None, mu: Optional[float] = None) -> float:
        """Calculate average particle number <N>."""
        T = T or self.T
        mu = mu or self.mu
        beta = 1 / (K_B * T)
        Xi = self.grand_partition_function(T, mu)
        weights = np.exp(-beta * (self.E - mu * self.N))
        return np.sum(self.N * weights) / Xi

    def grand_potential(self, T: Optional[float] = None, mu: Optional[float] = None) -> float:
        """Calculate grand potential Ω = -kT ln(Ξ)."""
        T = T or self.T
        mu = mu or self.mu
        return -K_B * T * np.log(self.grand_partition_function(T, mu))

    def pressure(self, V: float, T: Optional[float] = None, mu: Optional[float] = None) -> float:
        """Calculate pressure: P = -Ω/V."""
        return -self.grand_potential(T, mu) / V


class PartitionFunction(BaseClass):
    """
    Partition function calculator for various systems.

    Args:
        system_type: 'harmonic_oscillator', 'rotor', 'translation', 'electronic'
    """

    def __init__(self, system_type: str = 'harmonic_oscillator'):
        super().__init__()
        self.system_type = system_type

    def Z_harmonic_oscillator(self, T: float, omega: float) -> float:
        """
        Partition function for quantum harmonic oscillator.

        Z = 1 / (2 sinh(ℏω/2kT))
        """
        x = HBAR * omega / (2 * K_B * T)
        return 1 / (2 * np.sinh(x))

    def Z_classical_oscillator(self, T: float, omega: float) -> float:
        """Classical harmonic oscillator: Z = kT/(ℏω)."""
        return K_B * T / (HBAR * omega)

    def Z_rigid_rotor(self, T: float, B: float) -> float:
        """
        Partition function for rigid rotor.

        Z ≈ T / θ_rot where θ_rot = ℏ²/(2Ik_B)
        B is rotational constant in Hz.
        """
        theta_rot = H * B / K_B
        return T / theta_rot

    def Z_translation_3D(self, T: float, V: float, m: float) -> float:
        """
        Translational partition function.

        Z = V / Λ³ where Λ = h/√(2πmkT)
        """
        Lambda = H / np.sqrt(2 * np.pi * m * K_B * T)
        return V / Lambda**3


class EquipartitionTheorem(BaseClass):
    """
    Equipartition theorem for classical systems.

    Each quadratic degree of freedom contributes kT/2 to average energy.
    """

    def __init__(self):
        super().__init__()

    def average_energy(
        self,
        T: float,
        translational_dof: int = 3,
        rotational_dof: int = 0,
        vibrational_dof: int = 0
    ) -> float:
        """
        Calculate average energy per particle.

        Translation: 3 DOF → (3/2)kT
        Rotation: linear 2, nonlinear 3 DOF
        Vibration: each mode gives kT (kinetic + potential)
        """
        dof = translational_dof + rotational_dof + 2 * vibrational_dof
        return 0.5 * dof * K_B * T

    def heat_capacity(
        self,
        N: int,
        translational_dof: int = 3,
        rotational_dof: int = 0,
        vibrational_dof: int = 0
    ) -> float:
        """Calculate heat capacity C_V = dE/dT."""
        dof = translational_dof + rotational_dof + 2 * vibrational_dof
        return 0.5 * dof * N * K_B


# ==============================================================================
# Phase 4.5: Quantum Statistics
# ==============================================================================

class BoseEinsteinDistribution(BaseClass):
    """
    Bose-Einstein distribution for bosons.

    <n> = 1 / (exp((ε - μ)/kT) - 1)

    Args:
        T: Temperature (K)
        mu: Chemical potential (J)
    """

    def __init__(self, T: float, mu: float = 0.0):
        super().__init__()
        self.T = T
        self.mu = mu

    def occupation(self, energy: float) -> float:
        """Calculate average occupation number."""
        if energy <= self.mu:
            return np.inf  # Singularity
        x = (energy - self.mu) / (K_B * self.T)
        return 1 / (np.exp(x) - 1 + 1e-30)

    def critical_temperature(self, n_density: float, m: float) -> float:
        """
        Calculate BEC critical temperature.

        T_c = (2πℏ²/mk) (n/ζ(3/2))^(2/3)
        """
        zeta_3_2 = 2.612  # Riemann zeta(3/2)
        return (2 * np.pi * HBAR**2 / (m * K_B)) * (n_density / zeta_3_2)**(2/3)

    def condensate_fraction(self, T: float, T_c: float) -> float:
        """Calculate BEC condensate fraction below T_c."""
        if T >= T_c:
            return 0.0
        return 1 - (T / T_c)**1.5


class FermiDiracDistribution(BaseClass):
    """
    Fermi-Dirac distribution for fermions.

    f(ε) = 1 / (exp((ε - μ)/kT) + 1)

    Args:
        T: Temperature (K)
        mu: Chemical potential / Fermi energy (J)
    """

    def __init__(self, T: float, mu: float):
        super().__init__()
        self.T = T
        self.mu = mu
        self.E_F = mu  # Fermi energy at T=0

    def occupation(self, energy: float) -> float:
        """Calculate Fermi-Dirac occupation."""
        x = (energy - self.mu) / (K_B * self.T)
        if x > 100:
            return 0.0
        if x < -100:
            return 1.0
        return 1 / (np.exp(x) + 1)

    def fermi_energy_3D(self, n_density: float, m: float) -> float:
        """
        Calculate 3D Fermi energy at T=0.

        E_F = (ℏ²/2m)(3π²n)^(2/3)
        """
        return (HBAR**2 / (2 * m)) * (3 * np.pi**2 * n_density)**(2/3)

    def fermi_temperature(self) -> float:
        """Calculate Fermi temperature T_F = E_F / k_B."""
        return self.E_F / K_B

    def electronic_heat_capacity(self, T: float, N: int) -> float:
        """
        Calculate electronic heat capacity (Sommerfeld expansion).

        C_el = (π²/3) N k_B (k_B T / E_F)
        """
        return (np.pi**2 / 3) * N * K_B * (K_B * T / self.E_F)


class MaxwellBoltzmannDistribution(BaseClass):
    """
    Maxwell-Boltzmann distribution (classical limit).

    f(v) = 4π n (m/2πkT)^(3/2) v² exp(-mv²/2kT)

    Args:
        T: Temperature (K)
        m: Particle mass (kg)
    """

    def __init__(self, T: float, m: float):
        super().__init__()
        self.T = T
        self.m = m

    def speed_distribution(self, v: float) -> float:
        """Calculate probability density for speed v."""
        factor = (self.m / (2 * np.pi * K_B * self.T))**1.5
        return 4 * np.pi * factor * v**2 * np.exp(-self.m * v**2 / (2 * K_B * self.T))

    def most_probable_speed(self) -> float:
        """Calculate most probable speed v_p = √(2kT/m)."""
        return np.sqrt(2 * K_B * self.T / self.m)

    def mean_speed(self) -> float:
        """Calculate mean speed <v> = √(8kT/πm)."""
        return np.sqrt(8 * K_B * self.T / (np.pi * self.m))

    def rms_speed(self) -> float:
        """Calculate RMS speed √<v²> = √(3kT/m)."""
        return np.sqrt(3 * K_B * self.T / self.m)

    def energy_distribution(self, E: float) -> float:
        """Calculate probability density for energy E."""
        factor = 2 * np.pi * (1 / (np.pi * K_B * self.T))**1.5
        return factor * np.sqrt(E) * np.exp(-E / (K_B * self.T))


class PhotonGas(BaseClass):
    """
    Photon gas (blackbody radiation).

    Planck distribution for photon occupation number.

    Args:
        T: Temperature (K)
    """

    def __init__(self, T: float):
        super().__init__()
        self.T = T
        self.sigma = 5.670374e-8  # Stefan-Boltzmann constant

    def planck_distribution(self, frequency: float) -> float:
        """
        Planck distribution: mean photon number per mode.

        n(ν) = 1 / (exp(hν/kT) - 1)
        """
        x = H * frequency / (K_B * self.T)
        return 1 / (np.exp(x) - 1 + 1e-30)

    def spectral_energy_density(self, frequency: float) -> float:
        """
        Spectral energy density u(ν).

        u(ν) = (8πhν³/c³) / (exp(hν/kT) - 1)
        """
        c = 3e8
        x = H * frequency / (K_B * self.T)
        return (8 * np.pi * H * frequency**3 / c**3) / (np.exp(x) - 1 + 1e-30)

    def total_energy_density(self) -> float:
        """Calculate total energy density u = aT⁴ where a = 4σ/c."""
        c = 3e8
        return 4 * self.sigma * self.T**4 / c

    def peak_frequency(self) -> float:
        """Calculate peak frequency (Wien's law): ν_max = 2.82 kT/h."""
        return 2.821 * K_B * self.T / H

    def stefan_boltzmann_power(self, area: float) -> float:
        """Calculate radiated power: P = σ A T⁴."""
        return self.sigma * area * self.T**4


class PhononGas(BaseClass):
    """
    Phonon gas for lattice vibrations.

    Args:
        T: Temperature (K)
        omega_D: Debye cutoff frequency (rad/s)
        N: Number of atoms
    """

    def __init__(self, T: float, omega_D: float, N: int):
        super().__init__()
        self.T = T
        self.omega_D = omega_D
        self.N = N
        self.theta_D = HBAR * omega_D / K_B  # Debye temperature

    def average_energy(self) -> float:
        """Calculate average phonon energy using Debye model."""
        x = self.theta_D / self.T
        D = self._debye_function(x)
        return 9 * self.N * K_B * self.T * (self.T / self.theta_D)**3 * D + \
               (9/8) * self.N * K_B * self.theta_D  # Zero-point energy

    def _debye_function(self, x: float, n_points: int = 100) -> float:
        """Compute Debye integral D(x) = (3/x³) ∫₀ˣ t³/(eᵗ-1) dt."""
        if x < 0.01:
            return 1.0
        t = np.linspace(1e-10, x, n_points)
        dt = t[1] - t[0]
        integrand = t**3 / (np.exp(t) - 1 + 1e-30)
        return 3 * np.sum(integrand) * dt / x**3

    def heat_capacity(self) -> float:
        """Calculate heat capacity using Debye model."""
        x = self.theta_D / self.T
        if x < 0.1:
            # High T limit: C = 3Nk
            return 3 * self.N * K_B
        D = self._debye_function(x)
        return 9 * self.N * K_B * (self.T / self.theta_D)**3 * D * \
               (4 - 3 * x / (np.exp(x) - 1 + 1e-30))


class DebyeModel(BaseClass):
    """
    Debye model for heat capacity of solids.

    Args:
        theta_D: Debye temperature (K)
        N: Number of atoms
    """

    def __init__(self, theta_D: float, N: int):
        super().__init__()
        self.theta_D = theta_D
        self.N = N

    def heat_capacity(self, T: float) -> float:
        """Calculate heat capacity C_V at temperature T."""
        if T < 1e-10:
            return 0.0

        x = self.theta_D / T

        if x < 0.1:
            # High T: classical limit
            return 3 * self.N * K_B

        if x > 20:
            # Low T: T³ law
            return (12 * np.pi**4 / 5) * self.N * K_B * (T / self.theta_D)**3

        # General case: numerical Debye integral
        return 9 * self.N * K_B * (T / self.theta_D)**3 * self._debye_cv_integral(x)

    def _debye_cv_integral(self, x: float, n_points: int = 100) -> float:
        """Compute Debye heat capacity integral."""
        t = np.linspace(1e-10, x, n_points)
        dt = t[1] - t[0]
        integrand = t**4 * np.exp(t) / (np.exp(t) - 1 + 1e-30)**2
        return np.sum(integrand) * dt


class EinsteinModel(BaseClass):
    """
    Einstein model for heat capacity (simpler than Debye).

    Assumes all oscillators have same frequency ω_E.

    Args:
        theta_E: Einstein temperature (K) = ℏω_E/k_B
        N: Number of atoms
    """

    def __init__(self, theta_E: float, N: int):
        super().__init__()
        self.theta_E = theta_E
        self.N = N

    def heat_capacity(self, T: float) -> float:
        """Calculate Einstein heat capacity."""
        if T < 1e-10:
            return 0.0

        x = self.theta_E / T
        return 3 * self.N * K_B * x**2 * np.exp(x) / (np.exp(x) - 1 + 1e-30)**2


# ==============================================================================
# Phase 4.6: Phase Transitions
# ==============================================================================

class IsingModel1D(BaseClass):
    """
    1D Ising model (exact solution).

    H = -J Σ s_i s_{i+1} - h Σ s_i

    Args:
        N: Number of spins
        J: Coupling constant (J > 0 ferromagnetic)
        h: External field
    """

    def __init__(self, N: int, J: float = 1.0, h: float = 0.0):
        super().__init__()
        self.N = N
        self.J = J
        self.h = h

    def partition_function(self, T: float) -> float:
        """
        Calculate partition function using transfer matrix.

        Z = λ₊ᴺ + λ₋ᴺ ≈ λ₊ᴺ for large N
        """
        beta = 1 / (K_B * T)

        # Transfer matrix eigenvalues
        x = np.exp(beta * self.J)
        y = np.exp(-beta * self.J)
        z = np.exp(beta * self.h)
        z_inv = np.exp(-beta * self.h)

        # λ± = e^βJ cosh(βh) ± sqrt(e^2βJ sinh²(βh) + e^-2βJ)
        cosh_bh = (z + z_inv) / 2
        sinh_bh = (z - z_inv) / 2

        lambda_plus = x * cosh_bh + np.sqrt(x**2 * sinh_bh**2 + y**2)
        lambda_minus = x * cosh_bh - np.sqrt(x**2 * sinh_bh**2 + y**2)

        return lambda_plus**self.N + lambda_minus**self.N

    def free_energy(self, T: float) -> float:
        """Calculate Helmholtz free energy per spin."""
        return -K_B * T * np.log(self.partition_function(T)) / self.N

    def magnetization(self, T: float) -> float:
        """
        Calculate magnetization per spin.

        m = sinh(βh) / sqrt(sinh²(βh) + e^(-4βJ))
        """
        if abs(self.h) < 1e-15:
            return 0.0  # No spontaneous magnetization in 1D

        beta = 1 / (K_B * T)
        sinh_bh = np.sinh(beta * self.h)
        return sinh_bh / np.sqrt(sinh_bh**2 + np.exp(-4 * beta * self.J))

    def correlation_length(self, T: float) -> float:
        """Calculate correlation length ξ = -1/ln(tanh(βJ))."""
        beta = 1 / (K_B * T)
        tanh_bJ = np.tanh(beta * self.J)
        if tanh_bJ >= 1:
            return np.inf
        return -1 / np.log(tanh_bJ)


class IsingModel2D(BaseClass):
    """
    2D Ising model using Monte Carlo simulation.

    Args:
        L: Linear system size (L × L lattice)
        J: Coupling constant
        h: External field
    """

    def __init__(self, L: int, J: float = 1.0, h: float = 0.0):
        super().__init__()
        self.L = L
        self.J = J
        self.h = h
        self.N = L * L

        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(L, L))

        # Critical temperature (Onsager)
        self.T_c = 2 * J / (K_B * np.log(1 + np.sqrt(2)))

    def energy(self) -> float:
        """Calculate total energy of current configuration."""
        E = 0.0
        for i in range(self.L):
            for j in range(self.L):
                s = self.spins[i, j]
                # Neighbors (periodic boundary)
                neighbors = (self.spins[(i+1) % self.L, j] +
                            self.spins[(i-1) % self.L, j] +
                            self.spins[i, (j+1) % self.L] +
                            self.spins[i, (j-1) % self.L])
                E -= self.J * s * neighbors / 2  # Divide by 2 to avoid double counting
                E -= self.h * s
        return E

    def magnetization(self) -> float:
        """Calculate magnetization per spin."""
        return np.sum(self.spins) / self.N

    def metropolis_step(self, T: float):
        """Perform one Metropolis Monte Carlo step."""
        beta = 1 / (K_B * T)

        for _ in range(self.N):
            # Random site
            i = np.random.randint(self.L)
            j = np.random.randint(self.L)

            s = self.spins[i, j]
            neighbors = (self.spins[(i+1) % self.L, j] +
                        self.spins[(i-1) % self.L, j] +
                        self.spins[i, (j+1) % self.L] +
                        self.spins[i, (j-1) % self.L])

            # Energy change if flip
            dE = 2 * s * (self.J * neighbors + self.h)

            # Metropolis criterion
            if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                self.spins[i, j] = -s

    def simulate(
        self,
        T: float,
        n_steps: int = 1000,
        n_equilibrate: int = 100
    ) -> Tuple[float, float]:
        """
        Run Monte Carlo simulation and return average energy and magnetization.
        """
        # Equilibration
        for _ in range(n_equilibrate):
            self.metropolis_step(T)

        # Measurement
        E_sum = 0.0
        M_sum = 0.0

        for _ in range(n_steps):
            self.metropolis_step(T)
            E_sum += self.energy()
            M_sum += abs(self.magnetization())

        return E_sum / n_steps, M_sum / n_steps


class LandauTheory(BaseClass):
    """
    Landau theory of phase transitions.

    Free energy expansion: F = F_0 + a(T-T_c)φ² + bφ⁴ + ...

    Args:
        T_c: Critical temperature (K)
        a0: Coefficient in a = a₀(T - T_c)
        b: Fourth-order coefficient
    """

    def __init__(self, T_c: float, a0: float = 1.0, b: float = 1.0):
        super().__init__()
        self.T_c = T_c
        self.a0 = a0
        self.b = b

    def a(self, T: float) -> float:
        """Calculate temperature-dependent quadratic coefficient."""
        return self.a0 * (T - self.T_c)

    def free_energy(self, phi: float, T: float) -> float:
        """Calculate Landau free energy."""
        return self.a(T) * phi**2 + self.b * phi**4

    def order_parameter(self, T: float) -> float:
        """Calculate equilibrium order parameter."""
        if T >= self.T_c:
            return 0.0
        else:
            # Minimize F: dF/dφ = 0 → φ = ±√(-a/2b)
            return np.sqrt(-self.a(T) / (2 * self.b))

    def susceptibility(self, T: float) -> float:
        """Calculate susceptibility χ = -1/a (for T > T_c)."""
        a = self.a(T)
        if T >= self.T_c:
            return -1 / a
        else:
            return -1 / (2 * a)  # Below T_c

    def specific_heat_discontinuity(self) -> float:
        """Calculate specific heat discontinuity at T_c."""
        return self.a0**2 / (2 * self.b)


class CriticalExponents(BaseClass):
    """
    Critical exponents for phase transitions.

    Standard exponents: α, β, γ, δ, η, ν

    Args:
        universality_class: 'mean_field', 'ising_2d', 'ising_3d', 'xy_3d', 'heisenberg_3d'
    """

    EXPONENTS = {
        'mean_field': {'alpha': 0, 'beta': 0.5, 'gamma': 1, 'delta': 3, 'eta': 0, 'nu': 0.5},
        'ising_2d': {'alpha': 0, 'beta': 0.125, 'gamma': 1.75, 'delta': 15, 'eta': 0.25, 'nu': 1},
        'ising_3d': {'alpha': 0.11, 'beta': 0.326, 'gamma': 1.237, 'delta': 4.79, 'eta': 0.036, 'nu': 0.630},
        'xy_3d': {'alpha': -0.015, 'beta': 0.348, 'gamma': 1.316, 'delta': 4.78, 'eta': 0.038, 'nu': 0.671},
        'heisenberg_3d': {'alpha': -0.12, 'beta': 0.365, 'gamma': 1.386, 'delta': 4.80, 'eta': 0.037, 'nu': 0.707},
    }

    def __init__(self, universality_class: str = 'mean_field'):
        super().__init__()

        if universality_class not in self.EXPONENTS:
            raise ValidationError(f"Unknown universality class: {universality_class}")

        self.exponents = self.EXPONENTS[universality_class]
        self.universality_class = universality_class

    def alpha(self) -> float:
        """Heat capacity exponent: C ~ |t|^(-α)."""
        return self.exponents['alpha']

    def beta(self) -> float:
        """Order parameter exponent: φ ~ |t|^β."""
        return self.exponents['beta']

    def gamma(self) -> float:
        """Susceptibility exponent: χ ~ |t|^(-γ)."""
        return self.exponents['gamma']

    def delta(self) -> float:
        """Critical isotherm exponent: h ~ |φ|^δ sign(φ)."""
        return self.exponents['delta']

    def eta(self) -> float:
        """Correlation function exponent: G(r) ~ r^(-(d-2+η))."""
        return self.exponents['eta']

    def nu(self) -> float:
        """Correlation length exponent: ξ ~ |t|^(-ν)."""
        return self.exponents['nu']

    def check_scaling_relations(self) -> dict:
        """Check standard scaling relations."""
        exp = self.exponents
        relations = {
            'Rushbrooke': exp['alpha'] + 2*exp['beta'] + exp['gamma'],  # = 2
            'Widom': exp['gamma'] - exp['beta']*(exp['delta'] - 1),     # = 0
            'Fisher': exp['gamma'] - exp['nu']*(2 - exp['eta']),        # = 0 (d=3)
            'Josephson': 3*exp['nu'] - 2 + exp['alpha'],                # = 0 (d=3)
        }
        return relations


# ==============================================================================
# Phase 4.7: Non-equilibrium Thermodynamics
# ==============================================================================

class BoltzmannEquation(BaseClass):
    """
    Boltzmann transport equation for kinetic theory.

    ∂f/∂t + v·∇f + F·∇_v f = (∂f/∂t)_coll

    Args:
        collision_term: Collision operator (callable)
    """

    def __init__(self, collision_term: Optional[Callable] = None):
        super().__init__()
        self.collision = collision_term or self._bgk_collision

    def _bgk_collision(
        self,
        f: np.ndarray,
        f_eq: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """BGK (relaxation time) approximation for collision term."""
        return -(f - f_eq) / tau

    def maxwell_boltzmann_equilibrium(
        self,
        v: np.ndarray,
        n: float,
        T: float,
        u: np.ndarray,
        m: float
    ) -> np.ndarray:
        """Calculate Maxwell-Boltzmann equilibrium distribution."""
        factor = n * (m / (2 * np.pi * K_B * T))**1.5
        v_rel = v - u
        return factor * np.exp(-m * np.sum(v_rel**2) / (2 * K_B * T))


class LangevinDynamics(BaseClass):
    """
    Langevin equation for Brownian motion.

    m dv/dt = -γv + F(x) + √(2γkT) ξ(t)

    Args:
        mass: Particle mass (kg)
        gamma: Friction coefficient (kg/s)
        T: Temperature (K)
        force_func: External force F(x)
    """

    def __init__(
        self,
        mass: float,
        gamma: float,
        T: float,
        force_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        super().__init__()

        self.m = mass
        self.gamma = gamma
        self.T = T
        self.force = force_func or (lambda x: np.zeros_like(x))

        # Noise strength
        self.D = K_B * T / gamma  # Diffusion coefficient
        self.noise_amplitude = np.sqrt(2 * gamma * K_B * T)

    def step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one Langevin dynamics step.

        Uses velocity Verlet with stochastic term.
        """
        # Random force
        xi = np.random.normal(0, 1, size=x.shape) * np.sqrt(dt)

        # Update velocity (half step)
        F = self.force(x)
        v_half = v + (F / self.m - self.gamma * v / self.m + \
                      self.noise_amplitude * xi / (self.m * np.sqrt(dt))) * dt / 2

        # Update position
        x_new = x + v_half * dt

        # Update velocity (full step)
        F_new = self.force(x_new)
        v_new = v_half + (F_new / self.m - self.gamma * v_half / self.m) * dt / 2

        return x_new, v_new

    def simulate(
        self,
        x0: ArrayLike,
        v0: ArrayLike,
        dt: float,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Langevin dynamics simulation."""
        x = np.array(x0)
        v = np.array(v0)

        x_history = [x.copy()]
        v_history = [v.copy()]

        for _ in range(n_steps):
            x, v = self.step(x, v, dt)
            x_history.append(x.copy())
            v_history.append(v.copy())

        return np.array(x_history), np.array(v_history)


class FokkerPlanckEquation(BaseClass):
    """
    Fokker-Planck equation for probability evolution.

    ∂P/∂t = -∇·(A P) + (1/2) ∇²(D P)

    where A is drift and D is diffusion coefficient.

    Args:
        drift: Drift vector A(x)
        diffusion: Diffusion coefficient D (scalar or tensor)
    """

    def __init__(
        self,
        drift: Callable[[np.ndarray], np.ndarray],
        diffusion: Union[float, Callable[[np.ndarray], float]] = 1.0
    ):
        super().__init__()

        self.A = drift
        self.D = diffusion if callable(diffusion) else (lambda x: diffusion)

    def stationary_solution_1d(
        self,
        x: np.ndarray,
        potential: Callable[[float], float],
        T: float
    ) -> np.ndarray:
        """
        Calculate stationary solution for potential system.

        P_eq(x) ∝ exp(-V(x)/kT)
        """
        V = np.array([potential(xi) for xi in x])
        P = np.exp(-V / (K_B * T))
        return P / np.trapz(P, x)  # Normalize


class FluctuationDissipation(BaseClass):
    """
    Fluctuation-dissipation relations.

    Connects response functions to equilibrium fluctuations.
    """

    def __init__(self, T: float):
        super().__init__()
        self.T = T

    def classical_fdt(
        self,
        chi_omega: complex,
        omega: float
    ) -> float:
        """
        Classical FDT: S(ω) = (2kT/ω) Im[χ(ω)]

        where S(ω) is spectral density of fluctuations.
        """
        return 2 * K_B * self.T * np.imag(chi_omega) / omega

    def quantum_fdt(
        self,
        chi_omega: complex,
        omega: float
    ) -> float:
        """
        Quantum FDT with zero-point fluctuations.

        S(ω) = ℏ coth(ℏω/2kT) Im[χ(ω)]
        """
        x = HBAR * omega / (2 * K_B * self.T)
        return HBAR * np.imag(chi_omega) / np.tanh(x)

    def einstein_relation(self, mobility: float) -> float:
        """Einstein relation: D = μ kT."""
        return mobility * K_B * self.T


class JarzynskiEquality(BaseClass):
    """
    Jarzynski equality relating work and free energy.

    <exp(-βW)> = exp(-βΔF)

    Args:
        T: Temperature (K)
    """

    def __init__(self, T: float):
        super().__init__()
        self.T = T
        self.beta = 1 / (K_B * T)

    def free_energy_difference(self, work_values: ArrayLike) -> float:
        """
        Estimate ΔF from work measurements.

        ΔF = -kT ln<exp(-βW)>
        """
        W = np.array(work_values)
        exp_avg = np.mean(np.exp(-self.beta * W))
        return -K_B * self.T * np.log(exp_avg)

    def second_law_bound(self, work_values: ArrayLike) -> Tuple[float, float]:
        """
        Check second law: <W> ≥ ΔF.

        Returns (average_work, free_energy_difference).
        """
        W_avg = np.mean(work_values)
        delta_F = self.free_energy_difference(work_values)
        return W_avg, delta_F


class CrooksRelation(BaseClass):
    """
    Crooks fluctuation theorem for time-reversal symmetry.

    P_F(W) / P_R(-W) = exp(β(W - ΔF))

    Args:
        T: Temperature (K)
    """

    def __init__(self, T: float):
        super().__init__()
        self.T = T
        self.beta = 1 / (K_B * T)

    def crossing_point(
        self,
        work_forward: ArrayLike,
        work_reverse: ArrayLike,
        n_bins: int = 50
    ) -> float:
        """
        Find crossing point of P_F(W) and P_R(-W), which gives ΔF.
        """
        W_F = np.array(work_forward)
        W_R = np.array(work_reverse)

        # Create histograms
        W_min = min(W_F.min(), -W_R.max())
        W_max = max(W_F.max(), -W_R.min())
        bins = np.linspace(W_min, W_max, n_bins)

        P_F, _ = np.histogram(W_F, bins=bins, density=True)
        P_R, _ = np.histogram(-W_R, bins=bins, density=True)

        # Find crossing point
        bin_centers = (bins[:-1] + bins[1:]) / 2
        diff = P_F - P_R

        # Find zero crossing
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:
                # Linear interpolation
                W_cross = bin_centers[i] - diff[i] * (bin_centers[i+1] - bin_centers[i]) / (diff[i+1] - diff[i])
                return W_cross

        return np.mean(work_forward)  # Fallback
