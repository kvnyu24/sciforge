"""
Advanced Quantum Field Theory module

This module implements advanced QFT concepts including:
- Renormalization group flow
- Lattice field theory
- Anomalies and topological effects
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass
from scipy import integrate


@dataclass
class RGFlow:
    """
    Renormalization Group Flow analysis.

    Implements the renormalization group equations for studying
    how coupling constants change with energy scale.

    Args:
        beta_functions: List of beta functions β_i(g) for each coupling
        coupling_names: Names of the coupling constants
    """
    beta_functions: List[Callable]
    coupling_names: List[str]

    def __post_init__(self):
        if len(self.beta_functions) != len(self.coupling_names):
            raise ValueError("Number of beta functions must match number of couplings")
        self._history = {'scale': [], 'couplings': []}

    def flow(
        self,
        initial_couplings: ArrayLike,
        mu_initial: float,
        mu_final: float,
        n_steps: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Compute RG flow from initial to final scale.

        Args:
            initial_couplings: Initial values of coupling constants
            mu_initial: Initial energy scale
            mu_final: Final energy scale
            n_steps: Number of integration steps

        Returns:
            Dictionary with 'scale' and coupling values
        """
        initial_couplings = np.asarray(initial_couplings)
        t_initial = np.log(mu_initial)
        t_final = np.log(mu_final)
        t_span = np.linspace(t_initial, t_final, n_steps)

        def rhs(t, g):
            return np.array([beta(g) for beta in self.beta_functions])

        result = integrate.solve_ivp(
            rhs,
            (t_initial, t_final),
            initial_couplings,
            t_eval=t_span,
            method='RK45'
        )

        scales = np.exp(result.t)
        couplings = result.y.T

        self._history['scale'] = scales
        self._history['couplings'] = couplings

        output = {'scale': scales}
        for i, name in enumerate(self.coupling_names):
            output[name] = couplings[:, i]

        return output

    def find_fixed_points(
        self,
        search_range: Tuple[float, float] = (-5, 5),
        n_grid: int = 20
    ) -> List[np.ndarray]:
        """
        Find fixed points where all beta functions vanish.

        Args:
            search_range: Range to search for fixed points
            n_grid: Number of grid points per dimension

        Returns:
            List of fixed point locations
        """
        from scipy.optimize import fsolve

        n_couplings = len(self.coupling_names)
        fixed_points = []

        # Create grid of initial guesses
        grids = [np.linspace(search_range[0], search_range[1], n_grid)
                 for _ in range(n_couplings)]

        def beta_vector(g):
            return np.array([beta(g) for beta in self.beta_functions])

        # For 1D case
        if n_couplings == 1:
            for g0 in grids[0]:
                try:
                    fp, info, ier, _ = fsolve(beta_vector, [g0], full_output=True)
                    if ier == 1 and np.abs(beta_vector(fp)).max() < 1e-8:
                        # Check if already found
                        is_new = True
                        for existing in fixed_points:
                            if np.allclose(fp, existing, rtol=1e-4):
                                is_new = False
                                break
                        if is_new:
                            fixed_points.append(fp)
                except:
                    pass
        else:
            # Multi-dimensional search
            from itertools import product
            for init in product(*grids):
                try:
                    fp, info, ier, _ = fsolve(beta_vector, init, full_output=True)
                    if ier == 1 and np.abs(beta_vector(fp)).max() < 1e-8:
                        is_new = True
                        for existing in fixed_points:
                            if np.allclose(fp, existing, rtol=1e-4):
                                is_new = False
                                break
                        if is_new:
                            fixed_points.append(fp)
                except:
                    pass

        return fixed_points


class BetaFunction:
    """
    Beta function calculations for common field theories.

    Provides standard beta functions for QED, QCD, and scalar theories.
    """

    @staticmethod
    def qed_one_loop(e: ArrayLike) -> np.ndarray:
        """
        One-loop QED beta function.

        β(e) = e³/(12π²)

        Args:
            e: Electric coupling (can be array)

        Returns:
            Beta function value
        """
        e = np.asarray(e)
        return e**3 / (12 * np.pi**2)

    @staticmethod
    def qcd_one_loop(g: ArrayLike, n_f: int = 6, n_c: int = 3) -> np.ndarray:
        """
        One-loop QCD beta function.

        β(g) = -g³/(16π²) * (11*N_c/3 - 2*N_f/3)

        Args:
            g: Strong coupling
            n_f: Number of quark flavors
            n_c: Number of colors

        Returns:
            Beta function value
        """
        g = np.asarray(g)
        b0 = (11 * n_c / 3 - 2 * n_f / 3)
        return -g**3 / (16 * np.pi**2) * b0

    @staticmethod
    def phi4_one_loop(lam: ArrayLike, d: int = 4) -> np.ndarray:
        """
        One-loop φ⁴ theory beta function.

        β(λ) = (d-4)λ + 3λ²/(16π²)  (in d dimensions)

        Args:
            lam: Quartic coupling
            d: Spacetime dimension

        Returns:
            Beta function value
        """
        lam = np.asarray(lam)
        return (d - 4) * lam + 3 * lam**2 / (16 * np.pi**2)

    @staticmethod
    def yukawa_one_loop(y: ArrayLike) -> np.ndarray:
        """
        One-loop Yukawa theory beta function.

        β(y) = y³/(16π²) * (5/2)

        Args:
            y: Yukawa coupling

        Returns:
            Beta function value
        """
        y = np.asarray(y)
        return 5 * y**3 / (32 * np.pi**2)


@dataclass
class AnomalousDimension:
    """
    Anomalous dimension calculations.

    Computes the anomalous scaling dimensions of operators
    in quantum field theory.

    Args:
        gamma_function: Function γ(g) giving anomalous dimension
        operator_name: Name of the operator
    """
    gamma_function: Callable
    operator_name: str = "φ"

    def __call__(self, coupling: float) -> float:
        """Compute anomalous dimension at given coupling."""
        return self.gamma_function(coupling)

    def scaling_dimension(self, coupling: float, classical_dim: float) -> float:
        """
        Compute full scaling dimension.

        Δ = Δ_classical + γ(g)

        Args:
            coupling: Coupling constant value
            classical_dim: Classical scaling dimension

        Returns:
            Full scaling dimension
        """
        return classical_dim + self.gamma_function(coupling)

    @staticmethod
    def phi4_field(lam: float) -> float:
        """
        Anomalous dimension of φ field in φ⁴ theory (one-loop).

        γ_φ = 0 at one-loop in d=4
        """
        return 0.0

    @staticmethod
    def qed_electron(alpha: float) -> float:
        """
        Anomalous dimension of electron field in QED (one-loop).

        γ_ψ = -α/(4π)
        """
        return -alpha / (4 * np.pi)


class FixedPointRG:
    """
    Analysis of RG fixed points and critical exponents.

    Args:
        beta_functions: List of beta functions
        coupling_names: Names of couplings
    """

    def __init__(self, beta_functions: List[Callable], coupling_names: List[str]):
        self.beta_functions = beta_functions
        self.coupling_names = coupling_names
        self.n_couplings = len(coupling_names)

    def stability_matrix(self, fixed_point: ArrayLike, eps: float = 1e-6) -> np.ndarray:
        """
        Compute stability matrix at a fixed point.

        M_ij = ∂β_i/∂g_j |_{g*}

        Args:
            fixed_point: Location of fixed point
            eps: Finite difference step size

        Returns:
            Stability matrix
        """
        fixed_point = np.asarray(fixed_point)
        n = self.n_couplings
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                g_plus = fixed_point.copy()
                g_minus = fixed_point.copy()
                g_plus[j] += eps
                g_minus[j] -= eps

                M[i, j] = (self.beta_functions[i](g_plus) -
                          self.beta_functions[i](g_minus)) / (2 * eps)

        return M

    def critical_exponents(self, fixed_point: ArrayLike) -> np.ndarray:
        """
        Compute critical exponents at fixed point.

        The critical exponents are eigenvalues of the stability matrix.
        Negative eigenvalues → relevant directions (UV unstable)
        Positive eigenvalues → irrelevant directions (IR unstable)

        Args:
            fixed_point: Location of fixed point

        Returns:
            Array of critical exponents
        """
        M = self.stability_matrix(fixed_point)
        eigenvalues = np.linalg.eigvals(M)
        return np.sort(np.real(eigenvalues))[::-1]

    def classify_fixed_point(self, fixed_point: ArrayLike) -> str:
        """
        Classify fixed point type.

        Args:
            fixed_point: Location of fixed point

        Returns:
            Classification string
        """
        exponents = self.critical_exponents(fixed_point)
        n_relevant = np.sum(exponents < 0)
        n_irrelevant = np.sum(exponents > 0)
        n_marginal = np.sum(np.abs(exponents) < 1e-10)

        if n_relevant == len(exponents):
            return "UV fixed point (all relevant)"
        elif n_irrelevant == len(exponents):
            return "IR fixed point (all irrelevant)"
        elif n_marginal > 0:
            return f"Mixed with {n_marginal} marginal direction(s)"
        else:
            return f"Saddle point ({n_relevant} relevant, {n_irrelevant} irrelevant)"


@dataclass
class CallanSymanzik:
    """
    Callan-Symanzik equation solver.

    Implements the Callan-Symanzik equation:
    [μ ∂/∂μ + β(g) ∂/∂g + n*γ(g)] Γ^(n) = 0

    Args:
        beta: Beta function
        gamma: Anomalous dimension function
    """
    beta: Callable
    gamma: Callable

    def green_function_scaling(
        self,
        n_external: int,
        coupling: float,
        scale_ratio: float
    ) -> float:
        """
        Compute scaling of n-point Green function.

        Args:
            n_external: Number of external legs
            coupling: Coupling constant
            scale_ratio: Ratio of scales μ'/μ

        Returns:
            Multiplicative factor for Green function
        """
        gamma_val = self.gamma(coupling)
        return scale_ratio ** (-n_external * gamma_val)

    def running_coupling(
        self,
        coupling_init: float,
        mu_init: float,
        mu_final: float,
        n_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for running coupling g(μ).

        Args:
            coupling_init: Initial coupling at mu_init
            mu_init: Initial scale
            mu_final: Final scale
            n_steps: Integration steps

        Returns:
            Tuple of (scales, couplings)
        """
        t_init = np.log(mu_init)
        t_final = np.log(mu_final)
        t_span = np.linspace(t_init, t_final, n_steps)

        def rhs(t, g):
            return self.beta(g)

        result = integrate.solve_ivp(
            rhs,
            (t_init, t_final),
            [coupling_init],
            t_eval=t_span
        )

        return np.exp(result.t), result.y[0]


class LatticeScalar:
    """
    Lattice scalar field theory.

    Implements φ⁴ theory on a lattice with action:
    S = Σ_x [½(∇φ)² + ½m²φ² + λφ⁴/4!]

    Args:
        lattice_size: Size of cubic lattice
        mass_squared: Bare mass squared
        coupling: Quartic coupling λ
        spacing: Lattice spacing a
    """

    def __init__(
        self,
        lattice_size: int,
        mass_squared: float = 1.0,
        coupling: float = 0.1,
        spacing: float = 1.0
    ):
        self.L = lattice_size
        self.m2 = mass_squared
        self.lam = coupling
        self.a = spacing

        # Initialize field configuration
        self.phi = np.random.randn(lattice_size, lattice_size, lattice_size)
        self._history = {'action': [], 'phi_squared': []}

    def action(self, phi: Optional[np.ndarray] = None) -> float:
        """
        Compute lattice action.

        Args:
            phi: Field configuration (uses self.phi if None)

        Returns:
            Action value
        """
        if phi is None:
            phi = self.phi

        # Kinetic term (nearest neighbor)
        kinetic = 0.0
        for mu in range(3):
            phi_shift = np.roll(phi, -1, axis=mu)
            kinetic += np.sum((phi - phi_shift)**2)
        kinetic *= 0.5 / self.a**2

        # Mass term
        mass = 0.5 * self.m2 * np.sum(phi**2)

        # Interaction term
        interaction = self.lam * np.sum(phi**4) / 24

        return (kinetic + mass + interaction) * self.a**3

    def local_action(self, x: Tuple[int, int, int], phi_new: float) -> float:
        """
        Compute local action change for Metropolis update.

        Args:
            x: Lattice site coordinates
            phi_new: Proposed new field value

        Returns:
            Action change ΔS
        """
        i, j, k = x
        phi_old = self.phi[i, j, k]

        # Neighbor sum
        neighbors = (
            self.phi[(i+1) % self.L, j, k] +
            self.phi[(i-1) % self.L, j, k] +
            self.phi[i, (j+1) % self.L, k] +
            self.phi[i, (j-1) % self.L, k] +
            self.phi[i, j, (k+1) % self.L] +
            self.phi[i, j, (k-1) % self.L]
        )

        # Local action change
        delta_kinetic = (phi_new**2 - phi_old**2) * 3 / self.a**2
        delta_kinetic -= (phi_new - phi_old) * neighbors / self.a**2
        delta_mass = 0.5 * self.m2 * (phi_new**2 - phi_old**2)
        delta_interaction = self.lam * (phi_new**4 - phi_old**4) / 24

        return (delta_kinetic + delta_mass + delta_interaction) * self.a**3

    def metropolis_sweep(self, delta: float = 1.0, beta: float = 1.0) -> float:
        """
        Perform one Metropolis sweep.

        Args:
            delta: Maximum update size
            beta: Inverse temperature

        Returns:
            Acceptance rate
        """
        accepted = 0
        total = self.L**3

        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    phi_old = self.phi[i, j, k]
                    phi_new = phi_old + delta * (2 * np.random.random() - 1)

                    dS = self.local_action((i, j, k), phi_new)

                    if dS < 0 or np.random.random() < np.exp(-beta * dS):
                        self.phi[i, j, k] = phi_new
                        accepted += 1

        return accepted / total

    def thermalize(
        self,
        n_sweeps: int = 100,
        delta: float = 1.0,
        beta: float = 1.0
    ) -> List[float]:
        """
        Thermalize the configuration.

        Args:
            n_sweeps: Number of sweeps
            delta: Update size
            beta: Inverse temperature

        Returns:
            List of acceptance rates
        """
        rates = []
        for _ in range(n_sweeps):
            rate = self.metropolis_sweep(delta, beta)
            rates.append(rate)
            self._history['action'].append(self.action())
            self._history['phi_squared'].append(np.mean(self.phi**2))

        return rates

    def correlator(self, r_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two-point correlator ⟨φ(0)φ(r)⟩.

        Args:
            r_max: Maximum distance to compute

        Returns:
            Tuple of (distances, correlator values)
        """
        if r_max is None:
            r_max = self.L // 2

        correlator = np.zeros(r_max + 1)
        counts = np.zeros(r_max + 1)

        phi_mean = np.mean(self.phi)
        phi_shifted = self.phi - phi_mean

        for dx in range(r_max + 1):
            for dy in range(r_max + 1):
                for dz in range(r_max + 1):
                    r = int(np.sqrt(dx**2 + dy**2 + dz**2))
                    if r <= r_max:
                        phi_roll = np.roll(np.roll(np.roll(
                            phi_shifted, dx, axis=0), dy, axis=1), dz, axis=2)
                        correlator[r] += np.mean(phi_shifted * phi_roll)
                        counts[r] += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        correlator /= counts

        return np.arange(r_max + 1), correlator


class LatticeGauge:
    """
    Lattice gauge theory (U(1) or SU(N)).

    Implements Wilson's lattice gauge theory with link variables.

    Args:
        lattice_size: Size of 4D hypercubic lattice
        group: Gauge group ('U1' or 'SU2')
        beta: Inverse coupling β = 1/g²
    """

    def __init__(
        self,
        lattice_size: int,
        group: str = 'U1',
        beta: float = 1.0
    ):
        self.L = lattice_size
        self.group = group
        self.beta = beta

        if group == 'U1':
            # U(1) links are phases
            self.links = np.random.uniform(0, 2*np.pi,
                                          (lattice_size, lattice_size,
                                           lattice_size, lattice_size, 4))
        elif group == 'SU2':
            # SU(2) links are quaternions (a₀, a₁, a₂, a₃) with |a|=1
            a = np.random.randn(lattice_size, lattice_size,
                               lattice_size, lattice_size, 4, 4)
            norms = np.sqrt(np.sum(a**2, axis=-1, keepdims=True))
            self.links = a / norms
        else:
            raise ValueError(f"Unknown gauge group: {group}")

        self._history = {'action': [], 'plaquette': []}

    def plaquette(self, x: Tuple[int, int, int, int], mu: int, nu: int) -> complex:
        """
        Compute plaquette U_μν(x) = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x).

        Args:
            x: Lattice site (4-tuple)
            mu, nu: Plaquette directions

        Returns:
            Plaquette value (complex for U(1), trace for SU(N))
        """
        i, j, k, t = x
        L = self.L

        if self.group == 'U1':
            # Product of phases
            theta1 = self.links[i, j, k, t, mu]
            theta2 = self.links[(i + (mu==0)) % L,
                               (j + (mu==1)) % L,
                               (k + (mu==2)) % L,
                               (t + (mu==3)) % L, nu]
            theta3 = self.links[(i + (nu==0)) % L,
                               (j + (nu==1)) % L,
                               (k + (nu==2)) % L,
                               (t + (nu==3)) % L, mu]
            theta4 = self.links[i, j, k, t, nu]

            return np.exp(1j * (theta1 + theta2 - theta3 - theta4))
        else:
            raise NotImplementedError("SU(2) plaquette not yet implemented")

    def wilson_action(self) -> float:
        """
        Compute Wilson gauge action.

        S = β Σ_x Σ_{μ<ν} [1 - Re(U_μν(x))]

        Returns:
            Wilson action value
        """
        action = 0.0
        L = self.L

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for t in range(L):
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                plaq = self.plaquette((i, j, k, t), mu, nu)
                                action += 1 - np.real(plaq)

        return self.beta * action

    def average_plaquette(self) -> float:
        """
        Compute average plaquette value.

        Returns:
            ⟨Re(U_μν)⟩ averaged over lattice
        """
        total = 0.0
        count = 0
        L = self.L

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for t in range(L):
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                total += np.real(self.plaquette((i, j, k, t), mu, nu))
                                count += 1

        return total / count


class WilsonLoop:
    """
    Wilson loop calculations for confinement analysis.

    W(C) = Tr[P exp(i∮_C A·dx)]

    Args:
        gauge_field: LatticeGauge object
    """

    def __init__(self, gauge_field: LatticeGauge):
        self.gauge = gauge_field

    def rectangular_loop(
        self,
        origin: Tuple[int, int, int, int],
        R: int,
        T: int,
        spatial_dir: int = 0
    ) -> complex:
        """
        Compute R×T Wilson loop.

        Args:
            origin: Starting point
            R: Spatial extent
            T: Temporal extent
            spatial_dir: Spatial direction (0, 1, or 2)

        Returns:
            Wilson loop value
        """
        if self.gauge.group != 'U1':
            raise NotImplementedError("Only U(1) Wilson loops implemented")

        L = self.gauge.L
        i, j, k, t = origin
        mu = spatial_dir
        nu = 3  # time direction

        total_phase = 0.0

        # Bottom edge (spatial direction)
        for r in range(R):
            idx = [i, j, k, t]
            idx[mu] = (idx[mu] + r) % L
            total_phase += self.gauge.links[idx[0], idx[1], idx[2], idx[3], mu]

        # Right edge (temporal direction)
        idx = [i, j, k, t]
        idx[mu] = (idx[mu] + R) % L
        for s in range(T):
            total_phase += self.gauge.links[idx[0], idx[1], idx[2], idx[3], nu]
            idx[nu] = (idx[nu] + 1) % L

        # Top edge (spatial direction, backward)
        idx = [i, j, k, t]
        idx[nu] = (idx[nu] + T) % L
        for r in range(R):
            total_phase -= self.gauge.links[idx[0], idx[1], idx[2], idx[3], mu]
            idx[mu] = (idx[mu] + 1) % L

        # Left edge (temporal direction, backward)
        idx = [i, j, k, t]
        for s in range(T):
            total_phase -= self.gauge.links[idx[0], idx[1], idx[2], idx[3], nu]
            idx[nu] = (idx[nu] + 1) % L

        return np.exp(1j * total_phase)

    def static_potential(
        self,
        R_max: int,
        T: int,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract static quark potential from Wilson loops.

        V(R) = -lim_{T→∞} (1/T) ln⟨W(R,T)⟩

        Args:
            R_max: Maximum spatial separation
            T: Temporal extent
            n_samples: Number of origin samples

        Returns:
            Tuple of (R values, V(R) values)
        """
        L = self.gauge.L
        R_vals = np.arange(1, R_max + 1)
        W_avg = np.zeros(R_max)

        for R_idx, R in enumerate(R_vals):
            W_sum = 0.0
            for _ in range(n_samples):
                origin = tuple(np.random.randint(0, L, 4))
                W_sum += np.abs(self.rectangular_loop(origin, R, T))
            W_avg[R_idx] = W_sum / n_samples

        # Avoid log of zero
        W_avg = np.maximum(W_avg, 1e-10)
        V = -np.log(W_avg) / T

        return R_vals, V


@dataclass
class PlaquetteAction:
    """
    Plaquette action and improved actions.

    Implements standard Wilson and Symanzik-improved actions.
    """

    @staticmethod
    def wilson(plaquette_sum: float, beta: float) -> float:
        """Standard Wilson action."""
        return beta * plaquette_sum

    @staticmethod
    def symanzik(
        plaquette_sum: float,
        rectangle_sum: float,
        beta: float,
        c1: float = -1/12
    ) -> float:
        """
        Symanzik-improved action with O(a²) corrections removed.

        S = β[(1-8c₁)Σ plaquettes + c₁ Σ rectangles]

        Args:
            plaquette_sum: Sum over plaquettes
            rectangle_sum: Sum over 1×2 rectangles
            beta: Inverse coupling
            c1: Improvement coefficient

        Returns:
            Improved action value
        """
        return beta * ((1 - 8*c1) * plaquette_sum + c1 * rectangle_sum)


class ChiralAnomaly:
    """
    Chiral anomaly calculations.

    Implements the axial anomaly ∂_μ j^μ_5 = (α/2π) F_μν F̃^μν
    """

    def __init__(self, alpha: float = 1/137):
        """
        Args:
            alpha: Fine structure constant
        """
        self.alpha = alpha

    def anomaly_coefficient(self, n_flavors: int = 1) -> float:
        """
        Compute anomaly coefficient.

        For N_f flavors of fermions with charge Q:
        A = N_f * Q² * α/(2π)

        Args:
            n_flavors: Number of fermion flavors

        Returns:
            Anomaly coefficient
        """
        return n_flavors * self.alpha / (2 * np.pi)

    def divergence(
        self,
        E_field: ArrayLike,
        B_field: ArrayLike,
        n_flavors: int = 1
    ) -> float:
        """
        Compute anomalous divergence ∂_μ j^μ_5.

        Args:
            E_field: Electric field 3-vector
            B_field: Magnetic field 3-vector
            n_flavors: Number of flavors

        Returns:
            Anomalous divergence value
        """
        E = np.asarray(E_field)
        B = np.asarray(B_field)

        # F·F̃ = -4 E·B
        FF_dual = -4 * np.dot(E, B)

        return self.anomaly_coefficient(n_flavors) * FF_dual

    def index_theorem(self, instanton_number: int, n_flavors: int = 1) -> int:
        """
        Atiyah-Singer index theorem.

        n_+ - n_- = 2 * N_f * Q  (topological charge)

        Args:
            instanton_number: Instanton number Q
            n_flavors: Number of flavors

        Returns:
            Index (difference in zero modes)
        """
        return 2 * n_flavors * instanton_number


class Instanton:
    """
    Instanton solutions in gauge theory.

    Implements the BPST instanton solution for SU(2) gauge theory.

    Args:
        rho: Instanton size
        center: Instanton center location
    """

    def __init__(self, rho: float = 1.0, center: ArrayLike = None):
        self.rho = rho
        self.center = np.zeros(4) if center is None else np.asarray(center)

    def profile(self, x: ArrayLike) -> float:
        """
        Instanton profile function f(r).

        f(r) = r² / (r² + ρ²)

        Args:
            x: 4D position

        Returns:
            Profile value
        """
        x = np.asarray(x)
        r2 = np.sum((x - self.center)**2)
        return r2 / (r2 + self.rho**2)

    def gauge_field(self, x: ArrayLike, mu: int) -> np.ndarray:
        """
        BPST instanton gauge field A_μ(x).

        A_μ^a = η^a_μν (x-x₀)_ν f(r) / r²

        where η is the 't Hooft symbol.

        Args:
            x: 4D position
            mu: Lorentz index

        Returns:
            SU(2) gauge field (3-component)
        """
        x = np.asarray(x)
        y = x - self.center
        r2 = np.sum(y**2)

        if r2 < 1e-10:
            return np.zeros(3)

        f = self.profile(x)

        # 't Hooft symbols (self-dual)
        eta = np.zeros((3, 4, 4))
        # η^1
        eta[0, 0, 1] = eta[0, 2, 3] = 1
        eta[0, 1, 0] = eta[0, 3, 2] = -1
        # η^2
        eta[1, 0, 2] = eta[1, 3, 1] = 1
        eta[1, 2, 0] = eta[1, 1, 3] = -1
        # η^3
        eta[2, 0, 3] = eta[2, 1, 2] = 1
        eta[2, 3, 0] = eta[2, 2, 1] = -1

        A = np.zeros(3)
        for a in range(3):
            for nu in range(4):
                A[a] += eta[a, mu, nu] * y[nu]

        return A * f / r2

    def action(self) -> float:
        """
        Instanton action.

        S = 8π²/g² for one instanton (topological)

        Returns:
            Classical action (in units of 8π²/g²)
        """
        return 1.0  # One instanton has Q=1

    def topological_charge(self) -> int:
        """Topological charge of instanton."""
        return 1


class ThetaTerm:
    """
    Theta term in gauge theory.

    Implements the CP-violating θ-term:
    S_θ = (θ/32π²) ∫ d⁴x F_μν F̃^μν

    Args:
        theta: Theta angle
    """

    def __init__(self, theta: float):
        self.theta = theta

    def action_contribution(self, topological_charge: float) -> float:
        """
        Compute θ-term contribution to action.

        S_θ = θ * Q where Q = (1/32π²) ∫ F F̃

        Args:
            topological_charge: Instanton number

        Returns:
            Action contribution
        """
        return self.theta * topological_charge

    def vacuum_energy(self, chi: float) -> float:
        """
        Vacuum energy density from θ-term.

        E(θ) = χ(1 - cos θ) where χ is topological susceptibility

        Args:
            chi: Topological susceptibility

        Returns:
            Vacuum energy density
        """
        return chi * (1 - np.cos(self.theta))

    def neutron_edm(self, f_pi: float = 93e6, m_pi: float = 140e6) -> float:
        """
        Estimate neutron EDM from θ-term.

        d_n ≈ θ * e * m_q / (4π² f_π²) * ln(f_π/m_π)

        Args:
            f_pi: Pion decay constant (eV)
            m_pi: Pion mass (eV)

        Returns:
            Neutron EDM estimate (e·cm)
        """
        e = 1.6e-19  # Coulomb
        m_q = 5e6  # Light quark mass ~ 5 MeV

        d_n = self.theta * e * m_q / (4 * np.pi**2 * f_pi**2)
        d_n *= np.log(f_pi / m_pi)
        d_n *= 1e-13  # Convert to e·cm

        return d_n


class TopologicalCharge:
    """
    Topological charge calculations.

    Q = (1/32π²) ∫ d⁴x ε^μνρσ F_μν F_ρσ
    """

    def __init__(self, gauge_field: Optional[LatticeGauge] = None):
        self.gauge = gauge_field

    @staticmethod
    def from_field_strength(F: np.ndarray, volume: float) -> float:
        """
        Compute topological charge from field strength tensor.

        Args:
            F: Field strength tensor F_μν (4x4 antisymmetric)
            volume: Spacetime volume

        Returns:
            Topological charge
        """
        # Dual field strength
        F_dual = np.zeros_like(F)
        eps = np.zeros((4, 4, 4, 4))
        eps[0, 1, 2, 3] = eps[1, 2, 3, 0] = eps[2, 3, 0, 1] = eps[3, 0, 1, 2] = 1
        eps[1, 0, 2, 3] = eps[0, 2, 3, 1] = eps[2, 3, 1, 0] = eps[3, 1, 0, 2] = -1
        # ... (full Levi-Civita)

        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        F_dual[mu, nu] += 0.5 * eps[mu, nu, rho, sigma] * F[rho, sigma]

        # F·F̃ = Tr(F_μν F̃^μν)
        FF_dual = np.sum(F * F_dual)

        return FF_dual * volume / (32 * np.pi**2)

    def lattice_charge(self) -> float:
        """
        Compute topological charge on lattice.

        Uses plaquette definition.

        Returns:
            Lattice topological charge
        """
        if self.gauge is None:
            raise ValueError("No gauge field set")

        if self.gauge.group != 'U1':
            raise NotImplementedError("Only U(1) implemented")

        L = self.gauge.L
        Q = 0.0

        # Sum imaginary part of plaquettes
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for t in range(L):
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                plaq = self.gauge.plaquette((i, j, k, t), mu, nu)
                                Q += np.imag(np.log(plaq))

        return Q / (2 * np.pi)


__all__ = [
    'RGFlow',
    'BetaFunction',
    'AnomalousDimension',
    'FixedPointRG',
    'CallanSymanzik',
    'LatticeScalar',
    'LatticeGauge',
    'WilsonLoop',
    'PlaquetteAction',
    'ChiralAnomaly',
    'Instanton',
    'ThetaTerm',
    'TopologicalCharge',
]
