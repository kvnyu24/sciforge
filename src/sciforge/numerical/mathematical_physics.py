"""
Mathematical Physics Module

Implements Green's functions, special functions, and integral transforms
commonly used in physics.

Classes:
- GreenFunction1D: 1D Green's function solver
- GreenFunction3D: 3D Green's function solver
- RetardedGreen: Causal propagator
- SpectralGreen: Spectral representation of Green's functions
- BesselFunctions: Bessel J, Y, I, K functions
- LegendrePolynomials: Legendre P, Q functions
- LaguerrePolynomials: Laguerre and associated Laguerre
- HypergeometricFunction: Generalized hypergeometric functions
- EllipticIntegrals: Complete and incomplete elliptic integrals
- LaplaceTransform: s-domain analysis
- HilbertTransform: Analytic signal computation
- HankelTransform: Cylindrical symmetry transforms
- MellinTransform: Scale-invariant analysis
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Union
from scipy import special
from scipy.integrate import quad, trapezoid
from scipy.fft import fft, ifft, fftfreq
from ..core.base import BaseClass


class GreenFunction1D(BaseClass):
    """
    1D Green's function solver for linear differential operators.

    Solves L[G(x, x')] = δ(x - x') with boundary conditions.

    Common operators:
    - Laplacian: d²/dx² → G(x, x') = -|x - x'|/2
    - Helmholtz: d²/dx² + k² → G = exp(ik|x-x'|)/(2ik)

    Args:
        operator: Type of operator ('laplacian', 'helmholtz', 'diffusion')
        domain: (x_min, x_max) spatial domain
        bc_type: Boundary condition type ('dirichlet', 'neumann', 'periodic')
    """

    def __init__(self, operator: str = 'laplacian',
                 domain: Tuple[float, float] = (0, 1),
                 bc_type: str = 'dirichlet',
                 **params):
        super().__init__()
        self.operator = operator
        self.domain = domain
        self.bc_type = bc_type
        self.params = params

        # Extract common parameters
        self.k = params.get('k', 1.0)  # Wave number for Helmholtz
        self.D = params.get('D', 1.0)  # Diffusion coefficient

    def free_space(self, x: float, x_prime: float) -> complex:
        """
        Free-space Green's function (infinite domain).

        Args:
            x: Field point
            x_prime: Source point

        Returns:
            Green's function value
        """
        r = abs(x - x_prime)

        if self.operator == 'laplacian':
            # -d²G/dx² = δ(x - x') → G = -|x - x'|/2
            return -r / 2

        elif self.operator == 'helmholtz':
            # (-d²/dx² - k²)G = δ → G = exp(ik|x|)/(2ik)
            k = self.k
            if r < 1e-10:
                return 1j / (2 * k)  # Regularized at origin
            return np.exp(1j * k * r) / (2j * k)

        elif self.operator == 'diffusion':
            # (d/dt - D d²/dx²)G = δ at t > 0
            # This is time-dependent; return steady-state
            return -r / (2 * self.D)

        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def with_boundaries(self, x: float, x_prime: float) -> complex:
        """
        Green's function with boundary conditions using method of images.
        """
        a, b = self.domain
        L = b - a

        if self.bc_type == 'dirichlet':
            # Dirichlet: G = 0 at boundaries
            # Use eigenfunction expansion
            G = 0.0
            for n in range(1, 100):  # Truncate series
                kn = n * np.pi / L
                if self.operator == 'laplacian':
                    G += (2/L) * np.sin(kn * (x - a)) * np.sin(kn * (x_prime - a)) / kn**2
                elif self.operator == 'helmholtz':
                    denom = kn**2 - self.k**2
                    if abs(denom) > 1e-10:
                        G += (2/L) * np.sin(kn * (x - a)) * np.sin(kn * (x_prime - a)) / denom
            return G

        elif self.bc_type == 'neumann':
            # Neumann: dG/dn = 0 at boundaries
            G = 0.0
            for n in range(0, 100):
                kn = n * np.pi / L
                norm = L if n == 0 else L/2
                if self.operator == 'laplacian' and n > 0:
                    G += (1/norm) * np.cos(kn * (x - a)) * np.cos(kn * (x_prime - a)) / kn**2
                elif self.operator == 'helmholtz':
                    denom = kn**2 - self.k**2
                    if abs(denom) > 1e-10:
                        G += (1/norm) * np.cos(kn * (x - a)) * np.cos(kn * (x_prime - a)) / denom
            return G

        elif self.bc_type == 'periodic':
            # Periodic boundary conditions
            G = 0.0
            for n in range(-50, 51):
                if n == 0:
                    continue
                kn = 2 * np.pi * n / L
                if self.operator == 'laplacian':
                    G += (1/L) * np.exp(1j * kn * (x - x_prime)) / kn**2
                elif self.operator == 'helmholtz':
                    denom = kn**2 - self.k**2
                    if abs(denom) > 1e-10:
                        G += (1/L) * np.exp(1j * kn * (x - x_prime)) / denom
            return np.real(G)

        return self.free_space(x, x_prime)

    def solve(self, source: Callable[[float], float],
              x_grid: np.ndarray) -> np.ndarray:
        """
        Solve Lu = f using Green's function: u(x) = ∫G(x,x')f(x')dx'.

        Args:
            source: Source function f(x)
            x_grid: Points at which to evaluate solution

        Returns:
            Solution u(x) at grid points
        """
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 0.01
        solution = np.zeros(len(x_grid), dtype=complex)

        for i, x in enumerate(x_grid):
            # Numerical integration
            integrand = [self.with_boundaries(x, xp) * source(xp)
                        for xp in x_grid]
            solution[i] = trapezoid(integrand, x_grid)

        return np.real(solution) if np.allclose(np.imag(solution), 0) else solution


class GreenFunction3D(BaseClass):
    """
    3D Green's function for common operators.

    Args:
        operator: 'laplacian', 'helmholtz', or 'wave'
    """

    def __init__(self, operator: str = 'laplacian', **params):
        super().__init__()
        self.operator = operator
        self.k = params.get('k', 1.0)
        self.c = params.get('c', 1.0)  # Wave speed

    def free_space(self, r: ArrayLike, r_prime: ArrayLike) -> complex:
        """
        Free-space 3D Green's function.

        Args:
            r: Field point [x, y, z]
            r_prime: Source point [x', y', z']
        """
        r = np.asarray(r)
        r_prime = np.asarray(r_prime)
        R = np.linalg.norm(r - r_prime)

        if R < 1e-10:
            R = 1e-10  # Regularize

        if self.operator == 'laplacian':
            # -∇²G = δ → G = 1/(4πR)
            return 1 / (4 * np.pi * R)

        elif self.operator == 'helmholtz':
            # (-∇² - k²)G = δ → G = exp(ikR)/(4πR)
            return np.exp(1j * self.k * R) / (4 * np.pi * R)

        elif self.operator == 'poisson':
            # Same as Laplacian for electrostatics
            return 1 / (4 * np.pi * R)

        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def gradient(self, r: ArrayLike, r_prime: ArrayLike) -> np.ndarray:
        """Gradient of Green's function ∇G."""
        r = np.asarray(r)
        r_prime = np.asarray(r_prime)
        diff = r - r_prime
        R = np.linalg.norm(diff)

        if R < 1e-10:
            return np.zeros(3)

        if self.operator == 'laplacian':
            return -diff / (4 * np.pi * R**3)

        elif self.operator == 'helmholtz':
            k = self.k
            G = np.exp(1j * k * R) / (4 * np.pi * R)
            return diff / R * G * (1j * k - 1/R)

        return np.zeros(3)

    def solve_potential(self, charge_density: Callable,
                        field_points: np.ndarray,
                        source_points: np.ndarray,
                        weights: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation using Green's function.

        φ(r) = ∫ G(r, r') ρ(r') dV'

        Args:
            charge_density: Function ρ(r') or array of values
            field_points: Points to evaluate potential [n_field, 3]
            source_points: Source integration points [n_source, 3]
            weights: Integration weights for source points

        Returns:
            Potential at field points
        """
        potential = np.zeros(len(field_points), dtype=complex)

        for i, r in enumerate(field_points):
            for j, r_prime in enumerate(source_points):
                if callable(charge_density):
                    rho = charge_density(r_prime)
                else:
                    rho = charge_density[j]

                potential[i] += self.free_space(r, r_prime) * rho * weights[j]

        return np.real(potential) if np.allclose(np.imag(potential), 0) else potential


class RetardedGreen(BaseClass):
    """
    Retarded Green's function for wave equations.

    G_ret(r, t; r', t') = δ(t - t' - |r - r'|/c) / (4π|r - r'|)

    Enforces causality: G = 0 for t < t' + |r - r'|/c

    Args:
        c: Wave speed (m/s)
        dim: Spatial dimension (1, 2, or 3)
    """

    def __init__(self, c: float = 1.0, dim: int = 3):
        super().__init__()
        self.c = c
        self.dim = dim

    def evaluate(self, r: ArrayLike, t: float,
                 r_prime: ArrayLike, t_prime: float) -> float:
        """
        Evaluate retarded Green's function.

        Args:
            r: Field point
            t: Field time
            r_prime: Source point
            t_prime: Source time

        Returns:
            G_ret value (0 if acausal)
        """
        r = np.asarray(r).flatten()[:self.dim]
        r_prime = np.asarray(r_prime).flatten()[:self.dim]
        R = np.linalg.norm(r - r_prime)

        # Retarded time
        t_ret = t - t_prime - R / self.c

        # Causality
        if t_ret < 0:
            return 0.0

        if self.dim == 1:
            # G = c/2 * Θ(t - t' - |x - x'|/c)
            return self.c / 2 if t_ret >= 0 else 0.0

        elif self.dim == 2:
            # G = c/(2π) * Θ(t_ret) / √(c²t_ret² - R²) for R < ct_ret
            arg = self.c**2 * t_ret**2 - R**2
            if arg <= 0:
                return 0.0
            return self.c / (2 * np.pi * np.sqrt(arg))

        elif self.dim == 3:
            # G = δ(t_ret) / (4πR)
            # In practice, use narrow Gaussian approximation
            if R < 1e-10:
                R = 1e-10
            sigma = 0.01 / self.c  # Width of delta approximation
            return np.exp(-t_ret**2 / (2*sigma**2)) / (4 * np.pi * R * sigma * np.sqrt(2*np.pi))

        return 0.0

    def lienard_wiechert_potential(self, charge: float,
                                   trajectory: Callable[[float], np.ndarray],
                                   velocity: Callable[[float], np.ndarray],
                                   r: ArrayLike, t: float) -> Tuple[float, np.ndarray]:
        """
        Compute Liénard-Wiechert potentials for moving point charge.

        φ(r,t) = q / (4πε₀ κR_ret)
        A(r,t) = q v_ret / (4πε₀ c κR_ret)

        where κ = 1 - n̂·β, R_ret = |r - r_ret|

        Args:
            charge: Particle charge
            trajectory: r'(t) position function
            velocity: v(t) velocity function
            r: Field point
            t: Observation time

        Returns:
            (scalar_potential, vector_potential)
        """
        from scipy.optimize import brentq

        r = np.asarray(r)

        # Find retarded time by solving t - t_ret = |r - r'(t_ret)|/c
        def retarded_condition(t_ret):
            r_ret = trajectory(t_ret)
            return t - t_ret - np.linalg.norm(r - r_ret) / self.c

        # Search for retarded time
        try:
            t_ret = brentq(retarded_condition, t - 10, t - 1e-10)
        except ValueError:
            return 0.0, np.zeros(3)

        r_ret = trajectory(t_ret)
        v_ret = velocity(t_ret)
        R_vec = r - r_ret
        R = np.linalg.norm(R_vec)
        n_hat = R_vec / R

        beta = v_ret / self.c
        kappa = 1 - np.dot(n_hat, beta)

        # Potentials (Gaussian units, set 4πε₀ = 1)
        phi = charge / (kappa * R)
        A = charge * v_ret / (self.c * kappa * R)

        return phi, A


class SpectralGreen(BaseClass):
    """
    Spectral (frequency domain) representation of Green's functions.

    G(r, r'; ω) = Fourier transform of time-domain Green's function.

    Args:
        operator: Type of wave operator
        c: Wave speed
    """

    def __init__(self, operator: str = 'wave', c: float = 1.0):
        super().__init__()
        self.operator = operator
        self.c = c

    def frequency_domain_3d(self, r: ArrayLike, r_prime: ArrayLike,
                            omega: float) -> complex:
        """
        3D Green's function in frequency domain.

        For wave equation: G(ω) = exp(iωR/c) / (4πR)
        """
        r = np.asarray(r)
        r_prime = np.asarray(r_prime)
        R = np.linalg.norm(r - r_prime)

        if R < 1e-10:
            R = 1e-10

        k = omega / self.c
        return np.exp(1j * k * R) / (4 * np.pi * R)

    def spectral_density(self, omega: np.ndarray, R: float) -> np.ndarray:
        """
        Spectral density |G(ω)|² for fixed separation R.
        """
        return 1 / (4 * np.pi * R)**2 * np.ones_like(omega)

    def kramers_kronig(self, omega: np.ndarray,
                       chi_real: np.ndarray) -> np.ndarray:
        """
        Compute imaginary part from real part using Kramers-Kronig.

        χ''(ω) = -(1/π) P ∫ χ'(ω') / (ω' - ω) dω'
        """
        chi_imag = np.zeros_like(chi_real)
        d_omega = omega[1] - omega[0]

        for i, w in enumerate(omega):
            # Principal value integral
            integrand = chi_real / (omega - w + 1e-10)
            integrand[i] = 0  # Exclude pole
            chi_imag[i] = -trapezoid(integrand, omega) / np.pi

        return chi_imag


class BesselFunctions(BaseClass):
    """
    Bessel functions of various kinds.

    - J_n(x): Bessel function of first kind
    - Y_n(x): Bessel function of second kind (Neumann)
    - I_n(x): Modified Bessel of first kind
    - K_n(x): Modified Bessel of second kind
    - j_n(x): Spherical Bessel of first kind
    - y_n(x): Spherical Bessel of second kind

    Args:
        order: Order n of the Bessel function
    """

    def __init__(self, order: Union[int, float] = 0):
        super().__init__()
        self.order = order

    def J(self, x: ArrayLike) -> np.ndarray:
        """Bessel function of first kind J_n(x)."""
        return special.jv(self.order, x)

    def Y(self, x: ArrayLike) -> np.ndarray:
        """Bessel function of second kind Y_n(x)."""
        return special.yv(self.order, x)

    def I(self, x: ArrayLike) -> np.ndarray:
        """Modified Bessel function of first kind I_n(x)."""
        return special.iv(self.order, x)

    def K(self, x: ArrayLike) -> np.ndarray:
        """Modified Bessel function of second kind K_n(x)."""
        return special.kv(self.order, x)

    def j_spherical(self, x: ArrayLike) -> np.ndarray:
        """Spherical Bessel function of first kind j_n(x)."""
        return special.spherical_jn(int(self.order), np.asarray(x))

    def y_spherical(self, x: ArrayLike) -> np.ndarray:
        """Spherical Bessel function of second kind y_n(x)."""
        return special.spherical_yn(int(self.order), np.asarray(x))

    def H1(self, x: ArrayLike) -> np.ndarray:
        """Hankel function of first kind H^(1)_n = J_n + iY_n."""
        return special.hankel1(self.order, x)

    def H2(self, x: ArrayLike) -> np.ndarray:
        """Hankel function of second kind H^(2)_n = J_n - iY_n."""
        return special.hankel2(self.order, x)

    @staticmethod
    def zeros(n: int, k: int) -> np.ndarray:
        """First k zeros of J_n(x)."""
        return special.jn_zeros(n, k)

    @staticmethod
    def cylindrical_wave(n: int, k: float, r: float, phi: float,
                         outgoing: bool = True) -> complex:
        """
        Cylindrical wave solution: H_n(kr) * exp(in*phi).

        Args:
            n: Angular quantum number
            k: Wave number
            r: Radial distance
            phi: Azimuthal angle
            outgoing: Use H^(1) (outgoing) or H^(2) (incoming)
        """
        if outgoing:
            H = special.hankel1(n, k * r)
        else:
            H = special.hankel2(n, k * r)
        return H * np.exp(1j * n * phi)


class LegendrePolynomials(BaseClass):
    """
    Legendre polynomials and associated Legendre functions.

    P_l(x): Legendre polynomial of degree l
    P_l^m(x): Associated Legendre function
    Q_l(x): Legendre function of second kind

    Args:
        l: Degree
        m: Order (for associated functions), default 0
    """

    def __init__(self, l: int, m: int = 0):
        super().__init__()
        self.l = l
        self.m = m

    def P(self, x: ArrayLike) -> np.ndarray:
        """Legendre polynomial P_l(x)."""
        return special.eval_legendre(self.l, x)

    def P_associated(self, x: ArrayLike) -> np.ndarray:
        """Associated Legendre function P_l^m(x)."""
        return special.lpmv(self.m, self.l, x)

    def Q(self, x: ArrayLike) -> np.ndarray:
        """Legendre function of second kind Q_l(x)."""
        x = np.asarray(x)
        # Q_l can be computed from the hypergeometric representation
        # For now use recursion or numerical integration
        if self.l == 0:
            return 0.5 * np.log((1 + x) / (1 - x))
        elif self.l == 1:
            return x * 0.5 * np.log((1 + x) / (1 - x)) - 1
        else:
            # Use recurrence relation
            Q0 = 0.5 * np.log((1 + x) / (1 - x))
            Q1 = x * Q0 - 1
            for n in range(2, self.l + 1):
                Q2 = ((2*n - 1) * x * Q1 - (n - 1) * Q0) / n
                Q0, Q1 = Q1, Q2
            return Q1

    @staticmethod
    def spherical_harmonic(l: int, m: int,
                           theta: float, phi: float) -> complex:
        """
        Spherical harmonic Y_l^m(θ, φ).
        """
        return special.sph_harm(m, l, phi, theta)

    @staticmethod
    def expansion_coefficients(f: Callable[[float], float],
                               l_max: int, n_points: int = 100) -> np.ndarray:
        """
        Compute Legendre expansion coefficients.

        f(x) = Σ a_l P_l(x), a_l = (2l+1)/2 ∫_{-1}^1 f(x) P_l(x) dx
        """
        x = np.linspace(-1, 1, n_points)
        coeffs = np.zeros(l_max + 1)

        for l in range(l_max + 1):
            P_l = special.eval_legendre(l, x)
            f_vals = np.array([f(xi) for xi in x])
            coeffs[l] = (2*l + 1) / 2 * trapezoid(f_vals * P_l, x)

        return coeffs


class LaguerrePolynomials(BaseClass):
    """
    Laguerre and associated Laguerre polynomials.

    L_n(x): Laguerre polynomial
    L_n^α(x): Generalized (associated) Laguerre polynomial

    Important in quantum mechanics (hydrogen atom radial functions).

    Args:
        n: Degree
        alpha: Parameter for generalized Laguerre (default 0)
    """

    def __init__(self, n: int, alpha: float = 0):
        super().__init__()
        self.n = n
        self.alpha = alpha

    def L(self, x: ArrayLike) -> np.ndarray:
        """Laguerre polynomial L_n(x)."""
        return special.eval_laguerre(self.n, x)

    def L_generalized(self, x: ArrayLike) -> np.ndarray:
        """Generalized Laguerre polynomial L_n^α(x)."""
        return special.eval_genlaguerre(self.n, self.alpha, x)

    @staticmethod
    def hydrogen_radial(n: int, l: int, r: np.ndarray,
                        a0: float = 1.0) -> np.ndarray:
        """
        Hydrogen atom radial wave function R_nl(r).

        R_nl = √[(2/na₀)³ (n-l-1)!/(2n(n+l)!)] × (2r/na₀)^l × e^(-r/na₀) × L_{n-l-1}^{2l+1}(2r/na₀)
        """
        rho = 2 * r / (n * a0)

        # Normalization
        from math import factorial
        norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)))

        # Radial function
        L = special.eval_genlaguerre(n-l-1, 2*l+1, rho)

        return norm * rho**l * np.exp(-rho/2) * L


class HypergeometricFunction(BaseClass):
    """
    Hypergeometric functions.

    ₂F₁(a, b; c; z): Gauss hypergeometric
    ₁F₁(a; b; z): Confluent hypergeometric (Kummer's function)
    ₀F₁(; b; z): Bessel-type hypergeometric

    Args:
        a, b, c: Parameters (depends on type)
    """

    def __init__(self, a: float = 1, b: float = 1, c: float = 1):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def gauss_2F1(self, z: ArrayLike) -> np.ndarray:
        """Gauss hypergeometric function ₂F₁(a, b; c; z)."""
        return special.hyp2f1(self.a, self.b, self.c, z)

    def confluent_1F1(self, z: ArrayLike) -> np.ndarray:
        """Confluent hypergeometric (Kummer) ₁F₁(a; b; z)."""
        return special.hyp1f1(self.a, self.b, z)

    def U_tricomi(self, z: ArrayLike) -> np.ndarray:
        """Tricomi's confluent hypergeometric U(a, b, z)."""
        return special.hyperu(self.a, self.b, z)

    @staticmethod
    def coulomb_wave(l: int, eta: float, rho: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Coulomb wave functions F_l(η, ρ) and G_l(η, ρ).

        Solutions to Coulomb scattering problem.
        """
        rho = np.asarray(rho)
        F = np.zeros_like(rho)
        G = np.zeros_like(rho)

        for i, r in enumerate(rho):
            result = special.coulomb_wave(l, eta, r)
            F[i] = result[0]
            G[i] = result[1]

        return F, G


class EllipticIntegrals(BaseClass):
    """
    Elliptic integrals of the first, second, and third kinds.

    K(m): Complete elliptic integral of first kind
    E(m): Complete elliptic integral of second kind
    Π(n, m): Complete elliptic integral of third kind
    F(φ, m): Incomplete elliptic integral of first kind
    E(φ, m): Incomplete elliptic integral of second kind

    Note: Using convention where m = k² (modulus squared).

    Args:
        m: Modulus squared (0 ≤ m ≤ 1)
    """

    def __init__(self, m: float = 0.5):
        super().__init__()
        self.m = m

    def K_complete(self, m: Optional[float] = None) -> float:
        """
        Complete elliptic integral of first kind.

        K(m) = ∫₀^{π/2} dθ / √(1 - m sin²θ)
        """
        if m is None:
            m = self.m
        return special.ellipk(m)

    def E_complete(self, m: Optional[float] = None) -> float:
        """
        Complete elliptic integral of second kind.

        E(m) = ∫₀^{π/2} √(1 - m sin²θ) dθ
        """
        if m is None:
            m = self.m
        return special.ellipe(m)

    def F_incomplete(self, phi: float, m: Optional[float] = None) -> float:
        """
        Incomplete elliptic integral of first kind.

        F(φ, m) = ∫₀^φ dθ / √(1 - m sin²θ)
        """
        if m is None:
            m = self.m
        return special.ellipkinc(phi, m)

    def E_incomplete(self, phi: float, m: Optional[float] = None) -> float:
        """
        Incomplete elliptic integral of second kind.

        E(φ, m) = ∫₀^φ √(1 - m sin²θ) dθ
        """
        if m is None:
            m = self.m
        return special.ellipeinc(phi, m)

    def Pi_complete(self, n: float, m: Optional[float] = None) -> float:
        """
        Complete elliptic integral of third kind.

        Π(n, m) = ∫₀^{π/2} dθ / [(1 - n sin²θ)√(1 - m sin²θ)]
        """
        if m is None:
            m = self.m

        # Numerical integration
        def integrand(theta):
            s2 = np.sin(theta)**2
            return 1 / ((1 - n*s2) * np.sqrt(1 - m*s2))

        result, _ = quad(integrand, 0, np.pi/2)
        return result

    @staticmethod
    def pendulum_period(L: float, theta_max: float, g: float = 9.81) -> float:
        """
        Exact period of simple pendulum using elliptic integrals.

        T = 4√(L/g) K(sin²(θ_max/2))
        """
        m = np.sin(theta_max / 2)**2
        return 4 * np.sqrt(L / g) * special.ellipk(m)


class LaplaceTransform(BaseClass):
    """
    Laplace transform for s-domain analysis.

    F(s) = ∫₀^∞ f(t) e^{-st} dt

    Args:
        func: Time-domain function f(t)
    """

    def __init__(self, func: Optional[Callable[[float], float]] = None):
        super().__init__()
        self.func = func

    def forward(self, s: complex, t_max: float = 100,
                n_points: int = 1000) -> complex:
        """
        Numerical Laplace transform.

        Args:
            s: Complex frequency
            t_max: Integration upper limit
            n_points: Number of integration points
        """
        if self.func is None:
            raise ValueError("No function set")

        t = np.linspace(0, t_max, n_points)
        dt = t[1] - t[0]

        f_vals = np.array([self.func(ti) for ti in t])
        integrand = f_vals * np.exp(-s * t)

        return trapezoid(integrand, t)

    def inverse_bromwich(self, F: Callable[[complex], complex],
                         t: float, sigma: float = 1.0,
                         n_points: int = 1000) -> float:
        """
        Numerical inverse Laplace transform using Bromwich integral.

        f(t) = (1/2πi) ∫_{σ-i∞}^{σ+i∞} F(s) e^{st} ds

        Args:
            F: s-domain function
            t: Time point
            sigma: Real part of integration contour
            n_points: Number of integration points
        """
        omega_max = 100
        omega = np.linspace(-omega_max, omega_max, n_points)
        d_omega = omega[1] - omega[0]

        integrand = np.array([F(sigma + 1j*w) * np.exp((sigma + 1j*w) * t)
                             for w in omega])

        return np.real(trapezoid(integrand, omega)) / (2 * np.pi)

    @staticmethod
    def exponential(alpha: float, s: complex) -> complex:
        """Laplace transform of e^{αt}: L{e^{αt}} = 1/(s-α)."""
        return 1 / (s - alpha)

    @staticmethod
    def sine(omega: float, s: complex) -> complex:
        """Laplace transform of sin(ωt): L{sin(ωt)} = ω/(s² + ω²)."""
        return omega / (s**2 + omega**2)

    @staticmethod
    def cosine(omega: float, s: complex) -> complex:
        """Laplace transform of cos(ωt): L{cos(ωt)} = s/(s² + ω²)."""
        return s / (s**2 + omega**2)

    @staticmethod
    def step(s: complex) -> complex:
        """Laplace transform of unit step: L{u(t)} = 1/s."""
        return 1 / s

    @staticmethod
    def delta(s: complex) -> complex:
        """Laplace transform of delta: L{δ(t)} = 1."""
        return 1.0


class HilbertTransform(BaseClass):
    """
    Hilbert transform for analytic signal computation.

    H{f}(t) = (1/π) P ∫_{-∞}^{∞} f(τ)/(t - τ) dτ

    The analytic signal is: z(t) = f(t) + i H{f}(t)

    Args:
        signal: Input signal array
        dt: Time step
    """

    def __init__(self, signal: Optional[ArrayLike] = None, dt: float = 1.0):
        super().__init__()
        if signal is not None:
            self.signal = np.asarray(signal)
        else:
            self.signal = None
        self.dt = dt

    def transform(self, signal: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Compute Hilbert transform using FFT.
        """
        if signal is None:
            signal = self.signal
        signal = np.asarray(signal)

        N = len(signal)
        # FFT
        F = fft(signal)

        # Frequency multiplier for Hilbert transform
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 0
            h[1:N//2] = 1
            h[N//2] = 0
            h[N//2+1:] = -1
        else:
            h[0] = 0
            h[1:(N+1)//2] = 1
            h[(N+1)//2:] = -1

        return np.real(ifft(-1j * h * F))

    def analytic_signal(self, signal: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Compute analytic signal z(t) = f(t) + i H{f}(t).
        """
        if signal is None:
            signal = self.signal
        signal = np.asarray(signal)

        hilbert = self.transform(signal)
        return signal + 1j * hilbert

    def instantaneous_amplitude(self, signal: Optional[ArrayLike] = None) -> np.ndarray:
        """Instantaneous amplitude (envelope): A(t) = |z(t)|."""
        z = self.analytic_signal(signal)
        return np.abs(z)

    def instantaneous_phase(self, signal: Optional[ArrayLike] = None) -> np.ndarray:
        """Instantaneous phase: φ(t) = arg(z(t))."""
        z = self.analytic_signal(signal)
        return np.unwrap(np.angle(z))

    def instantaneous_frequency(self, signal: Optional[ArrayLike] = None) -> np.ndarray:
        """Instantaneous frequency: ω(t) = dφ/dt."""
        phase = self.instantaneous_phase(signal)
        return np.gradient(phase, self.dt) / (2 * np.pi)


class HankelTransform(BaseClass):
    """
    Hankel transform for functions with cylindrical symmetry.

    F_n(k) = ∫₀^∞ f(r) J_n(kr) r dr

    Args:
        order: Order n of Bessel function (default 0)
    """

    def __init__(self, order: int = 0):
        super().__init__()
        self.order = order

    def forward(self, f: Callable[[float], float],
                k: ArrayLike, r_max: float = 100,
                n_points: int = 1000) -> np.ndarray:
        """
        Numerical Hankel transform.

        Args:
            f: Radial function f(r)
            k: Wave numbers to evaluate at
            r_max: Integration upper limit
            n_points: Number of integration points
        """
        k = np.asarray(k)
        result = np.zeros_like(k, dtype=float)

        r = np.linspace(0, r_max, n_points)
        dr = r[1] - r[0]

        f_vals = np.array([f(ri) for ri in r])

        for i, ki in enumerate(k):
            J_n = special.jv(self.order, ki * r)
            result[i] = trapezoid(f_vals * J_n * r, r)

        return result

    def inverse(self, F: Callable[[float], float],
                r: ArrayLike, k_max: float = 100,
                n_points: int = 1000) -> np.ndarray:
        """
        Inverse Hankel transform (same form for integer orders).

        f(r) = ∫₀^∞ F(k) J_n(kr) k dk
        """
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)

        k = np.linspace(0, k_max, n_points)
        dk = k[1] - k[0]

        F_vals = np.array([F(ki) for ki in k])

        for i, ri in enumerate(r):
            J_n = special.jv(self.order, k * ri)
            result[i] = trapezoid(F_vals * J_n * k, k)

        return result

    @staticmethod
    def gaussian(k: ArrayLike, sigma: float = 1.0) -> np.ndarray:
        """
        Hankel transform of Gaussian: f(r) = exp(-r²/2σ²).

        H₀{exp(-r²/2σ²)} = σ² exp(-k²σ²/2)
        """
        k = np.asarray(k)
        return sigma**2 * np.exp(-k**2 * sigma**2 / 2)


class MellinTransform(BaseClass):
    """
    Mellin transform for scale-invariant analysis.

    M{f}(s) = ∫₀^∞ x^{s-1} f(x) dx

    Related to Fourier transform by: M{f}(s) = F{f(e^t)}(is)

    Args:
        func: Input function f(x)
    """

    def __init__(self, func: Optional[Callable[[float], float]] = None):
        super().__init__()
        self.func = func

    def forward(self, s: complex, x_min: float = 1e-6,
                x_max: float = 100, n_points: int = 1000) -> complex:
        """
        Numerical Mellin transform.

        Args:
            s: Complex parameter
            x_min, x_max: Integration limits
            n_points: Number of integration points
        """
        if self.func is None:
            raise ValueError("No function set")

        # Use logarithmic spacing for better accuracy
        x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)

        f_vals = np.array([self.func(xi) for xi in x])
        integrand = x**(s - 1) * f_vals

        # Integration in log space
        return trapezoid(integrand, x)

    def inverse(self, F: Callable[[complex], complex],
                x: float, c: float = 0.5,
                T: float = 100, n_points: int = 1000) -> float:
        """
        Inverse Mellin transform.

        f(x) = (1/2πi) ∫_{c-i∞}^{c+i∞} x^{-s} F(s) ds

        Args:
            F: Mellin transform function
            x: Point to evaluate at
            c: Real part of integration contour
            T: Imaginary part limits
            n_points: Number of integration points
        """
        t = np.linspace(-T, T, n_points)
        dt = t[1] - t[0]

        integrand = np.array([x**(-c - 1j*ti) * F(c + 1j*ti) for ti in t])

        return np.real(trapezoid(integrand, t)) / (2 * np.pi)

    @staticmethod
    def power_law(s: complex, alpha: float) -> complex:
        """
        Mellin transform of x^{-α}: M{x^{-α}} = 1/(s - α) for Re(s) > α.
        """
        return 1 / (s - alpha)

    @staticmethod
    def exponential(s: complex) -> complex:
        """
        Mellin transform of e^{-x}: M{e^{-x}} = Γ(s).
        """
        return special.gamma(s)


# Export all classes
__all__ = [
    'GreenFunction1D',
    'GreenFunction3D',
    'RetardedGreen',
    'SpectralGreen',
    'BesselFunctions',
    'LegendrePolynomials',
    'LaguerrePolynomials',
    'HypergeometricFunction',
    'EllipticIntegrals',
    'LaplaceTransform',
    'HilbertTransform',
    'HankelTransform',
    'MellinTransform',
]
