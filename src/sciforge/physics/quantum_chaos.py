"""
Quantum Chaos and Random Matrix Theory module

This module implements:
- Random matrix ensembles (GOE, GUE, GSE)
- Level statistics and spectral correlations
- Quantum chaos diagnostics (OTOC, ETH)
- Quantum scarring
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass
from scipy import special, integrate, linalg


class GaussianEnsemble:
    """
    Gaussian random matrix ensembles.

    Implements GOE (β=1), GUE (β=2), and GSE (β=4) ensembles.

    Args:
        n: Matrix dimension
        beta: Dyson index (1=orthogonal, 2=unitary, 4=symplectic)
    """

    def __init__(self, n: int, beta: int = 1):
        if beta not in [1, 2, 4]:
            raise ValueError("beta must be 1 (GOE), 2 (GUE), or 4 (GSE)")

        self.n = n
        self.beta = beta
        self._history = {'eigenvalues': [], 'spacings': []}

    def sample(self) -> np.ndarray:
        """
        Generate a random matrix from the ensemble.

        Returns:
            Random Hermitian matrix
        """
        if self.beta == 1:
            # GOE: real symmetric
            A = np.random.randn(self.n, self.n)
            return (A + A.T) / (2 * np.sqrt(self.n))

        elif self.beta == 2:
            # GUE: complex Hermitian
            A = (np.random.randn(self.n, self.n) +
                 1j * np.random.randn(self.n, self.n))
            return (A + A.conj().T) / (2 * np.sqrt(2 * self.n))

        else:  # beta == 4
            # GSE: quaternionic self-dual
            # Use 2N x 2N complex representation
            A = (np.random.randn(2*self.n, 2*self.n) +
                 1j * np.random.randn(2*self.n, 2*self.n))

            # Build symplectic structure
            J = np.zeros((2*self.n, 2*self.n), dtype=complex)
            J[:self.n, self.n:] = np.eye(self.n)
            J[self.n:, :self.n] = -np.eye(self.n)

            # Make self-dual
            H = (A + J @ A.conj().T @ J.conj().T) / (2 * np.sqrt(4 * self.n))
            return H

    def eigenvalues(self, H: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute eigenvalues (sorted).

        Args:
            H: Matrix to diagonalize (samples new if None)

        Returns:
            Sorted eigenvalues
        """
        if H is None:
            H = self.sample()

        evals = np.linalg.eigvalsh(H)
        self._history['eigenvalues'].append(evals)
        return np.sort(evals)

    def level_spacings(
        self,
        eigenvalues: Optional[np.ndarray] = None,
        unfold: bool = True
    ) -> np.ndarray:
        """
        Compute level spacings.

        Args:
            eigenvalues: Eigenvalue array (samples new if None)
            unfold: Whether to unfold spectrum first

        Returns:
            Array of level spacings
        """
        if eigenvalues is None:
            eigenvalues = self.eigenvalues()

        if unfold:
            eigenvalues = self.unfold_spectrum(eigenvalues)

        spacings = np.diff(eigenvalues)
        self._history['spacings'].append(spacings)
        return spacings

    @staticmethod
    def unfold_spectrum(eigenvalues: np.ndarray, poly_order: int = 5) -> np.ndarray:
        """
        Unfold spectrum to unit mean spacing.

        Args:
            eigenvalues: Raw eigenvalues
            poly_order: Polynomial order for fitting

        Returns:
            Unfolded eigenvalues
        """
        # Fit cumulative density
        N = len(eigenvalues)
        cumulative = np.arange(1, N + 1)

        # Polynomial fit to staircase function
        coeffs = np.polyfit(eigenvalues, cumulative, poly_order)
        smooth_N = np.polyval(coeffs, eigenvalues)

        return smooth_N

    def density_of_states(
        self,
        E: np.ndarray,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Compute average density of states.

        Args:
            E: Energy points
            n_samples: Number of samples to average

        Returns:
            Density of states
        """
        dos = np.zeros_like(E)
        sigma = 0.1 / np.sqrt(self.n)  # Gaussian broadening

        for _ in range(n_samples):
            evals = self.eigenvalues()
            for ev in evals:
                dos += np.exp(-(E - ev)**2 / (2 * sigma**2))

        dos /= (n_samples * sigma * np.sqrt(2 * np.pi))
        return dos

    def semicircle_law(self, E: np.ndarray) -> np.ndarray:
        """
        Wigner semicircle law (large N limit).

        ρ(E) = (1/2π) √(4-E²) for |E| < 2

        Args:
            E: Energy points

        Returns:
            Semicircle density
        """
        rho = np.zeros_like(E)
        mask = np.abs(E) < 2
        rho[mask] = np.sqrt(4 - E[mask]**2) / (2 * np.pi)
        return rho


class WignerSurmise:
    """
    Wigner surmise for level spacing distributions.

    P(s) = a_β s^β exp(-b_β s²)

    where β is the Dyson index.
    """

    def __init__(self, beta: int = 1):
        if beta not in [1, 2, 4]:
            raise ValueError("beta must be 1, 2, or 4")
        self.beta = beta

    def _coefficients(self) -> Tuple[float, float]:
        """Get coefficients a_β and b_β."""
        if self.beta == 1:
            a = np.pi / 2
            b = np.pi / 4
        elif self.beta == 2:
            a = 32 / np.pi**2
            b = 4 / np.pi
        else:  # beta == 4
            a = 2**18 / (3**6 * np.pi**3)
            b = 64 / (9 * np.pi)
        return a, b

    def pdf(self, s: ArrayLike) -> np.ndarray:
        """
        Probability density function.

        Args:
            s: Level spacing values

        Returns:
            P(s) values
        """
        s = np.asarray(s)
        a, b = self._coefficients()
        return a * s**self.beta * np.exp(-b * s**2)

    def cdf(self, s: ArrayLike) -> np.ndarray:
        """
        Cumulative distribution function.

        Args:
            s: Level spacing values

        Returns:
            CDF values
        """
        s = np.asarray(s)
        a, b = self._coefficients()

        # Integrate numerically
        result = np.zeros_like(s)
        for i, si in enumerate(s):
            result[i], _ = integrate.quad(self.pdf, 0, si)
        return result

    def mean_spacing(self) -> float:
        """Expected mean spacing ⟨s⟩."""
        a, b = self._coefficients()
        return special.gamma((self.beta + 2) / 2) * a / (2 * b**((self.beta + 2) / 2))

    def variance(self) -> float:
        """Variance of spacing distribution."""
        a, b = self._coefficients()
        # ⟨s²⟩ - ⟨s⟩²
        s2_mean = special.gamma((self.beta + 3) / 2) * a / (2 * b**((self.beta + 3) / 2))
        s_mean = self.mean_spacing()
        return s2_mean - s_mean**2


class MarchenkoPastur:
    """
    Marchenko-Pastur distribution for Wishart matrices.

    ρ(λ) = (1/2πσ²) √[(λ₊-λ)(λ-λ₋)] / (λγ)

    Args:
        gamma: Aspect ratio M/N
        sigma: Variance of matrix elements
    """

    def __init__(self, gamma: float, sigma: float = 1.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.sigma = sigma

        # Eigenvalue bounds
        self.lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
        self.lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2

    def pdf(self, lam: ArrayLike) -> np.ndarray:
        """
        Probability density function.

        Args:
            lam: Eigenvalue points

        Returns:
            Density values
        """
        lam = np.asarray(lam)
        rho = np.zeros_like(lam)

        mask = (lam > self.lambda_minus) & (lam < self.lambda_plus)
        lam_m = lam[mask]

        sqrt_term = np.sqrt((self.lambda_plus - lam_m) * (lam_m - self.lambda_minus))
        rho[mask] = sqrt_term / (2 * np.pi * self.sigma**2 * self.gamma * lam_m)

        # Point mass at zero if gamma > 1
        # (handled separately in applications)

        return rho

    def sample_wishart(self, M: int, N: int) -> np.ndarray:
        """
        Sample from Wishart ensemble W = X^T X / M.

        Args:
            M: Number of rows
            N: Number of columns

        Returns:
            Wishart matrix eigenvalues
        """
        X = np.random.randn(M, N) * self.sigma
        W = X.T @ X / M
        return np.sort(np.linalg.eigvalsh(W))


class TracyWidom:
    """
    Tracy-Widom distribution for largest eigenvalue fluctuations.

    Describes fluctuations of largest eigenvalue in GUE:
    P(λ_max < s) → F_2(s) as N → ∞

    where s = (λ_max - 2√N) N^{1/6}
    """

    def __init__(self, beta: int = 2):
        if beta not in [1, 2, 4]:
            raise ValueError("beta must be 1, 2, or 4")
        self.beta = beta

    def sample_largest_eigenvalue(
        self,
        N: int,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Sample scaled largest eigenvalues.

        Args:
            N: Matrix dimension
            n_samples: Number of samples

        Returns:
            Scaled largest eigenvalues
        """
        ensemble = GaussianEnsemble(N, self.beta)
        scaled_max = np.zeros(n_samples)

        for i in range(n_samples):
            evals = ensemble.eigenvalues()
            lambda_max = evals[-1]
            # Scale to Tracy-Widom variable
            scaled_max[i] = (lambda_max - 2) * N**(1/6) * np.sqrt(N)

        return scaled_max

    def approximate_pdf(self, s: ArrayLike) -> np.ndarray:
        """
        Approximate Tracy-Widom PDF using numerical fit.

        Args:
            s: Scaled eigenvalue positions

        Returns:
            Approximate PDF values
        """
        s = np.asarray(s)

        if self.beta == 2:
            # GUE Tracy-Widom: good approximation
            # Uses asymptotic forms
            pdf = np.zeros_like(s)

            # Left tail (s << 0): exp(-|s|^3/24)
            left = s < -2
            pdf[left] = np.exp(-np.abs(s[left])**3 / 24) / 8

            # Right tail (s >> 0): exp(-2s^{3/2}/3)
            right = s > 2
            pdf[right] = np.exp(-2 * s[right]**(3/2) / 3) * s[right]**(-1/4)

            # Central region: interpolate
            central = ~left & ~right
            # Approximate peak around s ≈ -1.77
            pdf[central] = 0.4 * np.exp(-(s[central] + 1.77)**2 / 2)

            return pdf
        else:
            raise NotImplementedError(f"TW-β={self.beta} approximation not implemented")


class LevelStatistics:
    """
    Level statistics analysis tools.

    Analyzes energy level distributions for signatures of
    quantum chaos vs integrability.
    """

    def __init__(self, eigenvalues: ArrayLike):
        self.eigenvalues = np.sort(np.asarray(eigenvalues))
        self.n_levels = len(self.eigenvalues)

    def spacing_ratio(self) -> np.ndarray:
        """
        Compute spacing ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

        Returns:
            Array of spacing ratios
        """
        spacings = np.diff(self.eigenvalues)
        n = len(spacings) - 1

        ratios = np.zeros(n)
        for i in range(n):
            s1, s2 = spacings[i], spacings[i+1]
            ratios[i] = min(s1, s2) / max(s1, s2)

        return ratios

    def mean_ratio(self) -> float:
        """
        Mean spacing ratio ⟨r⟩.

        ⟨r⟩ ≈ 0.386 for Poisson (integrable)
        ⟨r⟩ ≈ 0.530 for GOE (chaotic)
        ⟨r⟩ ≈ 0.603 for GUE

        Returns:
            Mean ratio value
        """
        return np.mean(self.spacing_ratio())

    def number_variance(self, L_max: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute number variance Σ²(L).

        Σ²(L) = ⟨(N(E, E+L) - L)²⟩

        Args:
            L_max: Maximum interval length
            n_points: Number of L values

        Returns:
            Tuple of (L values, Σ²(L) values)
        """
        unfolded = GaussianEnsemble.unfold_spectrum(self.eigenvalues)
        L_vals = np.linspace(0.1, L_max, n_points)
        sigma2 = np.zeros(n_points)

        for i, L in enumerate(L_vals):
            # Count levels in intervals of length L
            counts = []
            for start in unfolded[:-int(L) - 1]:
                n_in_interval = np.sum((unfolded >= start) & (unfolded < start + L))
                counts.append(n_in_interval)

            counts = np.array(counts)
            sigma2[i] = np.var(counts)

        return L_vals, sigma2

    def spectral_rigidity(self, L_max: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral rigidity Δ₃(L).

        Measures deviation from uniform level distribution.

        Args:
            L_max: Maximum interval length
            n_points: Number of L values

        Returns:
            Tuple of (L values, Δ₃(L) values)
        """
        unfolded = GaussianEnsemble.unfold_spectrum(self.eigenvalues)
        L_vals = np.linspace(0.1, L_max, n_points)
        delta3 = np.zeros(n_points)

        for i, L in enumerate(L_vals):
            # For each interval, fit best straight line
            deviations = []
            for start_idx in range(len(unfolded) - int(L) - 1):
                # Levels in interval
                mask = (unfolded >= unfolded[start_idx]) & \
                       (unfolded < unfolded[start_idx] + L)
                levels_in = unfolded[mask]

                if len(levels_in) > 2:
                    # Best fit line
                    x = np.arange(len(levels_in))
                    a, b = np.polyfit(x, levels_in, 1)
                    fit = a * x + b
                    deviation = np.mean((levels_in - fit)**2)
                    deviations.append(deviation)

            if deviations:
                delta3[i] = np.mean(deviations) / L

        return L_vals, delta3


class SpectralRigidity:
    """
    Theoretical predictions for spectral rigidity.
    """

    @staticmethod
    def poisson(L: ArrayLike) -> np.ndarray:
        """
        Poisson (integrable) spectral rigidity.

        Δ₃(L) = L/15

        Args:
            L: Interval length

        Returns:
            Spectral rigidity
        """
        return np.asarray(L) / 15

    @staticmethod
    def goe(L: ArrayLike) -> np.ndarray:
        """
        GOE spectral rigidity.

        Δ₃(L) ≈ (1/π²)[ln(2πL) + γ - 5/4 - π²/8] for large L

        Args:
            L: Interval length

        Returns:
            Spectral rigidity
        """
        L = np.asarray(L)
        gamma = 0.5772156649  # Euler-Mascheroni
        return (np.log(2 * np.pi * L) + gamma - 5/4 - np.pi**2 / 8) / np.pi**2

    @staticmethod
    def gue(L: ArrayLike) -> np.ndarray:
        """
        GUE spectral rigidity.

        Δ₃(L) ≈ (1/2π²)[ln(2πL) + γ - 5/4] for large L

        Args:
            L: Interval length

        Returns:
            Spectral rigidity
        """
        L = np.asarray(L)
        gamma = 0.5772156649
        return (np.log(2 * np.pi * L) + gamma - 5/4) / (2 * np.pi**2)


class ETHTest:
    """
    Eigenstate Thermalization Hypothesis (ETH) tests.

    Tests whether matrix elements of operators satisfy ETH:
    ⟨E_m|O|E_n⟩ = O(Ē)δ_{mn} + e^{-S(Ē)/2} f(Ē,ω) R_{mn}

    Args:
        hamiltonian: System Hamiltonian
        operator: Observable to test
    """

    def __init__(self, hamiltonian: np.ndarray, operator: np.ndarray):
        self.H = hamiltonian
        self.O = operator
        self.n = hamiltonian.shape[0]

        # Diagonalize
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian)

    def diagonal_elements(self) -> np.ndarray:
        """
        Compute diagonal matrix elements ⟨E_n|O|E_n⟩.

        Returns:
            Diagonal elements
        """
        O_diag = np.zeros(self.n)
        for n in range(self.n):
            v = self.eigenvectors[:, n]
            O_diag[n] = np.real(v.conj() @ self.O @ v)
        return O_diag

    def off_diagonal_variance(self, energy_window: float = 0.1) -> np.ndarray:
        """
        Compute variance of off-diagonal elements in energy windows.

        Args:
            energy_window: Width of energy window

        Returns:
            Variance as function of energy
        """
        E_mean = (self.eigenvalues.max() + self.eigenvalues.min()) / 2
        E_bins = np.arange(self.eigenvalues.min(), self.eigenvalues.max(), energy_window)

        variances = []
        for E in E_bins[:-1]:
            # Find states in window
            mask = (self.eigenvalues >= E) & (self.eigenvalues < E + energy_window)
            indices = np.where(mask)[0]

            if len(indices) > 1:
                # Compute off-diagonal elements
                off_diag = []
                for i, m in enumerate(indices):
                    for n in indices[i+1:]:
                        v_m = self.eigenvectors[:, m]
                        v_n = self.eigenvectors[:, n]
                        O_mn = v_m.conj() @ self.O @ v_n
                        off_diag.append(np.abs(O_mn)**2)

                if off_diag:
                    variances.append(np.mean(off_diag))
                else:
                    variances.append(0)
            else:
                variances.append(0)

        return np.array(variances)

    def microcanonical_average(self, energy: float, delta_E: float) -> float:
        """
        Compute microcanonical average in energy shell.

        Args:
            energy: Center energy
            delta_E: Shell width

        Returns:
            Microcanonical expectation value
        """
        mask = np.abs(self.eigenvalues - energy) < delta_E
        if not np.any(mask):
            return np.nan

        O_diag = self.diagonal_elements()
        return np.mean(O_diag[mask])

    def test_eth(
        self,
        energy_center: float,
        delta_E: float = 0.5
    ) -> Dict[str, float]:
        """
        Perform ETH test.

        Args:
            energy_center: Energy to test around
            delta_E: Energy window

        Returns:
            Dictionary with test results
        """
        # Get states in window
        mask = np.abs(self.eigenvalues - energy_center) < delta_E
        indices = np.where(mask)[0]

        if len(indices) < 2:
            return {'eth_score': np.nan, 'diagonal_variance': np.nan}

        O_diag = self.diagonal_elements()[mask]

        # Diagonal variance (should be small for ETH)
        diag_var = np.var(O_diag)

        # Off-diagonal statistics
        off_diag = []
        for i, m in enumerate(indices):
            for n in indices[i+1:]:
                v_m = self.eigenvectors[:, m]
                v_n = self.eigenvectors[:, n]
                O_mn = v_m.conj() @ self.O @ v_n
                off_diag.append(np.abs(O_mn)**2)

        off_diag_mean = np.mean(off_diag) if off_diag else 0

        # ETH score: ratio of diagonal to off-diagonal
        eth_score = diag_var / (off_diag_mean + 1e-10)

        return {
            'eth_score': eth_score,
            'diagonal_variance': diag_var,
            'off_diagonal_mean': off_diag_mean,
            'n_states': len(indices)
        }


class OTOCorrelator:
    """
    Out-of-Time-Order Correlator (OTOC).

    C(t) = ⟨[W(t), V]†[W(t), V]⟩

    Measures quantum chaos through operator spreading.

    Args:
        hamiltonian: System Hamiltonian
    """

    def __init__(self, hamiltonian: np.ndarray):
        self.H = hamiltonian
        self.n = hamiltonian.shape[0]

        # Diagonalize for efficient time evolution
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian)

    def evolve_operator(self, O: np.ndarray, t: float) -> np.ndarray:
        """
        Time evolve operator: O(t) = e^{iHt} O e^{-iHt}.

        Args:
            O: Operator to evolve
            t: Time

        Returns:
            Time-evolved operator
        """
        # O(t) = V e^{iEt} V† O V e^{-iEt} V†
        V = self.eigenvectors
        phases = np.exp(1j * self.eigenvalues * t)

        O_rotated = V.conj().T @ O @ V
        O_evolved = np.diag(phases) @ O_rotated @ np.diag(phases.conj())

        return V @ O_evolved @ V.conj().T

    def compute(
        self,
        W: np.ndarray,
        V: np.ndarray,
        times: ArrayLike,
        beta: float = 0.0
    ) -> np.ndarray:
        """
        Compute OTOC C(t) = ⟨[W(t), V]†[W(t), V]⟩.

        Args:
            W: First operator
            V: Second operator
            times: Time points
            beta: Inverse temperature (0 = infinite T)

        Returns:
            OTOC values at each time
        """
        times = np.asarray(times)
        otoc = np.zeros(len(times), dtype=complex)

        # Thermal density matrix
        if beta == 0:
            rho = np.eye(self.n) / self.n
        else:
            rho = np.diag(np.exp(-beta * self.eigenvalues))
            rho /= np.trace(rho)
            rho = self.eigenvectors @ rho @ self.eigenvectors.conj().T

        for i, t in enumerate(times):
            W_t = self.evolve_operator(W, t)
            commutator = W_t @ V - V @ W_t
            C = commutator.conj().T @ commutator
            otoc[i] = np.trace(rho @ C)

        return np.real(otoc)

    def lyapunov_from_otoc(
        self,
        W: np.ndarray,
        V: np.ndarray,
        t_max: float = 10.0,
        n_points: int = 100
    ) -> float:
        """
        Extract Lyapunov exponent from OTOC growth.

        For chaotic systems: C(t) ~ exp(2λ_L t)

        Args:
            W, V: Operators
            t_max: Maximum time
            n_points: Number of time points

        Returns:
            Estimated Lyapunov exponent
        """
        times = np.linspace(0.1, t_max, n_points)
        otoc = self.compute(W, V, times)

        # Fit exponential in initial growth regime
        # log(C) ~ 2λt
        log_otoc = np.log(otoc + 1e-10)

        # Find linear regime
        growth_mask = (otoc > 1e-8) & (otoc < 0.5 * otoc.max())
        if np.sum(growth_mask) > 5:
            coeffs = np.polyfit(times[growth_mask], log_otoc[growth_mask], 1)
            return coeffs[0] / 2  # λ = slope/2
        else:
            return 0.0


class QuantumScars:
    """
    Quantum scarring analysis.

    Quantum scars are eigenstates with anomalously high overlap
    with unstable periodic orbits.

    Args:
        hamiltonian: System Hamiltonian
    """

    def __init__(self, hamiltonian: np.ndarray):
        self.H = hamiltonian
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian)
        self.n = hamiltonian.shape[0]

    def overlap_with_state(self, target_state: np.ndarray) -> np.ndarray:
        """
        Compute overlap of eigenstates with target state.

        Args:
            target_state: State to compare (e.g., coherent state on orbit)

        Returns:
            Array of |⟨ψ|E_n⟩|² values
        """
        target = np.asarray(target_state)
        target = target / np.linalg.norm(target)

        overlaps = np.abs(self.eigenvectors.conj().T @ target)**2
        return overlaps

    def find_scars(
        self,
        target_state: np.ndarray,
        threshold: float = 3.0
    ) -> List[int]:
        """
        Find scarred eigenstates.

        Args:
            target_state: State on periodic orbit
            threshold: Multiple of mean overlap for scar detection

        Returns:
            Indices of scarred states
        """
        overlaps = self.overlap_with_state(target_state)
        mean_overlap = 1.0 / self.n  # Random state expectation

        scar_indices = np.where(overlaps > threshold * mean_overlap)[0]
        return list(scar_indices)

    def inverse_participation_ratio(self, basis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute IPR for each eigenstate.

        IPR = Σ |ψ_n(i)|⁴

        Small IPR = delocalized (thermal)
        Large IPR = localized (scarred)

        Args:
            basis: Basis to compute IPR in (default: computational basis)

        Returns:
            IPR for each eigenstate
        """
        if basis is None:
            # Computational basis
            psi = self.eigenvectors
        else:
            # Transform to new basis
            psi = basis.conj().T @ self.eigenvectors

        ipr = np.sum(np.abs(psi)**4, axis=0)
        return ipr

    def entanglement_entropy(
        self,
        subsystem_dim: int,
        eigenstate_idx: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute entanglement entropy of eigenstate(s).

        S = -Tr(ρ_A log ρ_A)

        Args:
            subsystem_dim: Dimension of subsystem A
            eigenstate_idx: Specific state index (None = all states)

        Returns:
            Entanglement entropy
        """
        if eigenstate_idx is not None:
            psi = self.eigenvectors[:, eigenstate_idx]
            return self._single_state_entropy(psi, subsystem_dim)
        else:
            entropies = np.zeros(self.n)
            for i in range(self.n):
                entropies[i] = self._single_state_entropy(
                    self.eigenvectors[:, i], subsystem_dim
                )
            return entropies

    def _single_state_entropy(self, psi: np.ndarray, dim_A: int) -> float:
        """Compute entropy for single state."""
        dim_B = len(psi) // dim_A

        # Reshape to bipartite form
        psi_matrix = psi.reshape(dim_A, dim_B)

        # Reduced density matrix
        rho_A = psi_matrix @ psi_matrix.conj().T

        # Eigenvalues for entropy
        eigs = np.linalg.eigvalsh(rho_A)
        eigs = eigs[eigs > 1e-15]

        return -np.sum(eigs * np.log(eigs))


__all__ = [
    'GaussianEnsemble',
    'WignerSurmise',
    'MarchenkoPastur',
    'TracyWidom',
    'LevelStatistics',
    'SpectralRigidity',
    'ETHTest',
    'OTOCorrelator',
    'QuantumScars',
]
