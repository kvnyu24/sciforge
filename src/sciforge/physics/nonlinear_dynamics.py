"""
Nonlinear Dynamics and Chaos Module

Implements tools for analyzing dynamical systems, chaos, and
nonlinear phenomena.

Classes:
- PhasePortrait: Phase space visualization and analysis
- FixedPointAnalysis: Equilibrium point classification
- StabilityAnalysis: Linear stability analysis
- BifurcationDiagram: Parameter-dependent behavior
- PoincareSection: Stroboscopic phase space sampling
- LyapunovExponent: Chaos quantification
- StrangeAttractor: Chaotic attractor analysis
- FractalDimension: Attractor dimensionality
- RecurrencePlot: Dynamical recurrence analysis
- LorenzSystem: Lorenz attractor
- RosslerSystem: Rössler attractor
- HenonMap: Discrete chaotic map
- LogisticMap: Period doubling cascade
- DoublePendulumChaos: Mechanical chaos example
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, List, Union
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist, squareform
from ..core.base import BaseClass


class FixedPointAnalysis(BaseClass):
    """
    Fixed point (equilibrium) analysis for dynamical systems.

    For dx/dt = f(x), finds x* where f(x*) = 0 and classifies stability.

    Args:
        dynamics: Function f(x) returning dx/dt, where x is state vector
        dim: Dimension of state space
    """

    def __init__(self, dynamics: Callable[[np.ndarray], np.ndarray], dim: int):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dim

    def find_fixed_point(self, x0: ArrayLike, tol: float = 1e-10) -> np.ndarray:
        """
        Find fixed point starting from initial guess.

        Args:
            x0: Initial guess for fixed point
            tol: Tolerance for convergence

        Returns:
            Fixed point location
        """
        x0 = np.asarray(x0)
        result = fsolve(self.dynamics, x0, full_output=True)
        x_fixed = result[0]

        # Verify it's actually a fixed point
        residual = np.linalg.norm(self.dynamics(x_fixed))
        if residual > tol:
            raise ValueError(f"Did not converge to fixed point (residual={residual})")

        return x_fixed

    def jacobian(self, x: ArrayLike, epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute Jacobian matrix at point x using finite differences.

        J_ij = ∂f_i/∂x_j
        """
        x = np.asarray(x)
        J = np.zeros((self.dim, self.dim))

        for j in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += epsilon
            x_minus[j] -= epsilon

            J[:, j] = (self.dynamics(x_plus) - self.dynamics(x_minus)) / (2 * epsilon)

        return J

    def classify_fixed_point(self, x_fixed: ArrayLike) -> dict:
        """
        Classify fixed point based on eigenvalues of Jacobian.

        Returns:
            Dictionary with classification information
        """
        J = self.jacobian(x_fixed)
        eigenvalues = np.linalg.eigvals(J)

        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        result = {
            'location': np.asarray(x_fixed),
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'real_parts': real_parts,
            'imag_parts': imag_parts,
        }

        # Classify based on eigenvalue real parts
        if np.all(real_parts < 0):
            if np.any(np.abs(imag_parts) > 1e-10):
                result['type'] = 'stable_spiral'
                result['stability'] = 'asymptotically_stable'
            else:
                result['type'] = 'stable_node'
                result['stability'] = 'asymptotically_stable'
        elif np.all(real_parts > 0):
            if np.any(np.abs(imag_parts) > 1e-10):
                result['type'] = 'unstable_spiral'
                result['stability'] = 'unstable'
            else:
                result['type'] = 'unstable_node'
                result['stability'] = 'unstable'
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            result['type'] = 'saddle'
            result['stability'] = 'unstable'
        elif np.all(np.abs(real_parts) < 1e-10):
            result['type'] = 'center'
            result['stability'] = 'marginally_stable'
        else:
            result['type'] = 'unknown'
            result['stability'] = 'unknown'

        return result

    def find_all_fixed_points(self, bounds: ArrayLike,
                              n_samples: int = 100) -> List[np.ndarray]:
        """
        Search for multiple fixed points by sampling initial conditions.

        Args:
            bounds: [[x_min, x_max], [y_min, y_max], ...] for each dimension
            n_samples: Number of random initial conditions to try

        Returns:
            List of unique fixed points found
        """
        bounds = np.asarray(bounds)
        fixed_points = []

        for _ in range(n_samples):
            # Random initial condition
            x0 = bounds[:, 0] + np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0])

            try:
                x_fixed = self.find_fixed_point(x0)

                # Check if this is a new fixed point
                is_new = True
                for xf in fixed_points:
                    if np.linalg.norm(x_fixed - xf) < 1e-6:
                        is_new = False
                        break

                if is_new and np.all(x_fixed >= bounds[:, 0]) and np.all(x_fixed <= bounds[:, 1]):
                    fixed_points.append(x_fixed)
            except (ValueError, RuntimeWarning):
                continue

        return fixed_points


class StabilityAnalysis(BaseClass):
    """
    Linear stability analysis for dynamical systems.

    Analyzes stability of fixed points and limit cycles.

    Args:
        dynamics: Function f(x) or f(x, t) returning dx/dt
        dim: State space dimension
    """

    def __init__(self, dynamics: Callable, dim: int):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dim

    def linear_stability_matrix(self, x0: ArrayLike,
                                 epsilon: float = 1e-8) -> np.ndarray:
        """Compute linearization matrix A where dx/dt ≈ A(x - x0)."""
        x0 = np.asarray(x0)
        A = np.zeros((self.dim, self.dim))

        f0 = self._eval_dynamics(x0)

        for j in range(self.dim):
            x_pert = x0.copy()
            x_pert[j] += epsilon
            A[:, j] = (self._eval_dynamics(x_pert) - f0) / epsilon

        return A

    def _eval_dynamics(self, x: np.ndarray, t: float = 0) -> np.ndarray:
        """Evaluate dynamics, handling both f(x) and f(x,t) signatures."""
        try:
            return np.asarray(self.dynamics(x, t))
        except TypeError:
            return np.asarray(self.dynamics(x))

    def growth_rates(self, x0: ArrayLike) -> np.ndarray:
        """
        Compute growth rates (real parts of eigenvalues) at x0.

        Negative = stable, Positive = unstable
        """
        A = self.linear_stability_matrix(x0)
        eigenvalues = np.linalg.eigvals(A)
        return np.real(eigenvalues)

    def is_stable(self, x0: ArrayLike) -> bool:
        """Check if fixed point is asymptotically stable."""
        return np.all(self.growth_rates(x0) < 0)

    def floquet_multipliers(self, periodic_orbit: np.ndarray,
                            period: float) -> np.ndarray:
        """
        Compute Floquet multipliers for periodic orbit stability.

        Args:
            periodic_orbit: Points along periodic orbit [n_points, dim]
            period: Orbit period

        Returns:
            Floquet multipliers (eigenvalues of monodromy matrix)
        """
        n_points = len(periodic_orbit)
        dt = period / n_points

        # Integrate variational equations around orbit
        M = np.eye(self.dim)  # Monodromy matrix

        for i in range(n_points):
            x = periodic_orbit[i]
            A = self.linear_stability_matrix(x)
            # Approximate matrix exponential for small dt
            M = (np.eye(self.dim) + A * dt) @ M

        return np.linalg.eigvals(M)


class PhasePortrait(BaseClass):
    """
    Phase portrait generation and visualization for 2D systems.

    Args:
        dynamics: Function f(x) returning dx/dt for 2D state x
        x_range: [x_min, x_max] for x-axis
        y_range: [y_min, y_max] for y-axis
    """

    def __init__(self, dynamics: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float]):
        super().__init__()
        self.dynamics = dynamics
        self.x_range = x_range
        self.y_range = y_range

    def vector_field(self, nx: int = 20, ny: int = 20) -> Tuple[np.ndarray, ...]:
        """
        Compute vector field on grid.

        Returns:
            X, Y, U, V: Grid coordinates and velocity components
        """
        x = np.linspace(self.x_range[0], self.x_range[1], nx)
        y = np.linspace(self.y_range[0], self.y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(ny):
            for j in range(nx):
                state = np.array([X[i, j], Y[i, j]])
                dstate = self.dynamics(state)
                U[i, j] = dstate[0]
                V[i, j] = dstate[1]

        return X, Y, U, V

    def nullclines(self, nx: int = 100, ny: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute nullclines (where dx/dt = 0 or dy/dt = 0).

        Returns:
            x_nullcline, y_nullcline: Arrays of (x, y) points
        """
        x = np.linspace(self.x_range[0], self.x_range[1], nx)
        y = np.linspace(self.y_range[0], self.y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # Evaluate dynamics on grid
        Fx = np.zeros_like(X)
        Fy = np.zeros_like(Y)

        for i in range(ny):
            for j in range(nx):
                state = np.array([X[i, j], Y[i, j]])
                dstate = self.dynamics(state)
                Fx[i, j] = dstate[0]
                Fy[i, j] = dstate[1]

        return Fx, Fy  # Contour at level 0 gives nullclines

    def trajectory(self, x0: ArrayLike, t_span: Tuple[float, float],
                   n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute trajectory starting from x0.

        Returns:
            t, trajectory: Time array and state trajectory
        """
        x0 = np.asarray(x0)
        t = np.linspace(t_span[0], t_span[1], n_points)

        def dynamics_wrapper(x, t):
            return self.dynamics(x)

        trajectory = odeint(dynamics_wrapper, x0, t)
        return t, trajectory

    def separatrices(self, saddle_point: ArrayLike,
                     epsilon: float = 0.01,
                     t_span: float = 10.0) -> List[np.ndarray]:
        """
        Compute separatrices (stable/unstable manifolds) of saddle point.

        Args:
            saddle_point: Location of saddle point
            epsilon: Small displacement for initial conditions
            t_span: Integration time

        Returns:
            List of separatrix trajectories
        """
        saddle = np.asarray(saddle_point)

        # Get eigenvectors of Jacobian
        fp_analysis = FixedPointAnalysis(self.dynamics, 2)
        J = fp_analysis.jacobian(saddle)
        eigenvalues, eigenvectors = np.linalg.eig(J)

        separatrices = []

        for i, ev in enumerate(eigenvalues):
            direction = np.real(eigenvectors[:, i])
            direction = direction / np.linalg.norm(direction)

            # Integrate forward and backward along each eigenvector
            for sign in [1, -1]:
                x0 = saddle + sign * epsilon * direction

                if np.real(ev) > 0:
                    # Unstable direction: integrate forward
                    _, traj = self.trajectory(x0, (0, t_span))
                else:
                    # Stable direction: integrate backward
                    _, traj = self.trajectory(x0, (0, -t_span))

                separatrices.append(traj)

        return separatrices


class BifurcationDiagram(BaseClass):
    """
    Bifurcation diagram computation for parameter-dependent systems.

    Args:
        dynamics: Function f(x, param) returning dx/dt
        dim: State space dimension
    """

    def __init__(self, dynamics: Callable[[np.ndarray, float], np.ndarray],
                 dim: int):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dim

    def compute_1d_map(self, param_range: Tuple[float, float],
                       n_params: int = 200,
                       x0: float = 0.5,
                       n_iterations: int = 1000,
                       n_last: int = 100,
                       map_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bifurcation diagram for 1D iterated map.

        x_{n+1} = f(x_n, param)

        Args:
            param_range: (param_min, param_max)
            n_params: Number of parameter values
            x0: Initial condition
            n_iterations: Total iterations per parameter
            n_last: Number of final values to record
            map_func: Optional 1D map function(x, param) -> x_next

        Returns:
            params, x_values: Parameter and corresponding x values
        """
        if map_func is None:
            map_func = lambda x, p: self.dynamics(np.array([x]), p)[0]

        params = np.linspace(param_range[0], param_range[1], n_params)
        all_params = []
        all_x = []

        for param in params:
            x = x0
            # Transient
            for _ in range(n_iterations - n_last):
                x = map_func(x, param)
            # Record
            for _ in range(n_last):
                x = map_func(x, param)
                all_params.append(param)
                all_x.append(x)

        return np.array(all_params), np.array(all_x)

    def track_fixed_point(self, param_range: Tuple[float, float],
                          x0_guess: ArrayLike,
                          n_params: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track fixed point as parameter varies (continuation).

        Returns:
            params, fixed_points, stability: Arrays of tracked values
        """
        params = np.linspace(param_range[0], param_range[1], n_params)
        fixed_points = []
        stability = []

        x_current = np.asarray(x0_guess)

        for param in params:
            # Find fixed point using previous as initial guess
            def f(x):
                return self.dynamics(x, param)

            try:
                x_fixed = fsolve(f, x_current, full_output=False)

                # Check stability
                fp_analysis = FixedPointAnalysis(f, self.dim)
                info = fp_analysis.classify_fixed_point(x_fixed)

                fixed_points.append(x_fixed.copy())
                stability.append(1 if info['stability'] == 'asymptotically_stable' else 0)

                x_current = x_fixed
            except:
                fixed_points.append(np.full(self.dim, np.nan))
                stability.append(np.nan)

        return params, np.array(fixed_points), np.array(stability)

    def find_bifurcation_points(self, param_range: Tuple[float, float],
                                x0_guess: ArrayLike,
                                n_params: int = 500) -> List[float]:
        """
        Find approximate bifurcation points where stability changes.

        Returns:
            List of parameter values where bifurcations occur
        """
        params, fixed_points, stability = self.track_fixed_point(
            param_range, x0_guess, n_params)

        bifurcation_points = []

        for i in range(1, len(stability)):
            if not np.isnan(stability[i]) and not np.isnan(stability[i-1]):
                if stability[i] != stability[i-1]:
                    bifurcation_points.append(0.5 * (params[i] + params[i-1]))

        return bifurcation_points


class PoincareSection(BaseClass):
    """
    Poincaré section (stroboscopic map) for continuous systems.

    Records state when trajectory crosses a surface of section.

    Args:
        dynamics: Function f(x, t) returning dx/dt
        dim: State space dimension
    """

    def __init__(self, dynamics: Callable[[np.ndarray, float], np.ndarray],
                 dim: int):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dim

    def plane_section(self, x0: ArrayLike, t_max: float,
                      section_index: int = 0,
                      section_value: float = 0.0,
                      direction: int = 1,
                      dt: float = 0.01) -> np.ndarray:
        """
        Compute Poincaré section on plane x[section_index] = section_value.

        Args:
            x0: Initial condition
            t_max: Maximum integration time
            section_index: Which state variable defines the section
            section_value: Value of that variable on the section
            direction: +1 for positive crossings, -1 for negative, 0 for both
            dt: Integration time step

        Returns:
            Array of section crossings [n_crossings, dim]
        """
        x0 = np.asarray(x0)
        t = np.arange(0, t_max, dt)

        def dynamics_wrapper(t, x):
            return self.dynamics(x, t)

        sol = solve_ivp(dynamics_wrapper, (0, t_max), x0,
                        t_eval=t, method='RK45')
        trajectory = sol.y.T

        crossings = []
        for i in range(1, len(trajectory)):
            x_prev = trajectory[i-1, section_index]
            x_curr = trajectory[i, section_index]

            # Check for crossing
            if direction >= 0 and x_prev < section_value <= x_curr:
                # Interpolate
                alpha = (section_value - x_prev) / (x_curr - x_prev)
                crossing = (1 - alpha) * trajectory[i-1] + alpha * trajectory[i]
                crossings.append(crossing)
            elif direction <= 0 and x_prev > section_value >= x_curr:
                alpha = (x_prev - section_value) / (x_prev - x_curr)
                crossing = (1 - alpha) * trajectory[i-1] + alpha * trajectory[i]
                crossings.append(crossing)

        return np.array(crossings) if crossings else np.array([]).reshape(0, self.dim)

    def stroboscopic_section(self, x0: ArrayLike, period: float,
                             n_periods: int = 1000) -> np.ndarray:
        """
        Compute stroboscopic Poincaré section (sample at fixed intervals).

        Args:
            x0: Initial condition
            period: Sampling period
            n_periods: Number of periods to sample

        Returns:
            Array of sampled states [n_periods, dim]
        """
        x0 = np.asarray(x0)
        t_eval = np.arange(n_periods + 1) * period

        def dynamics_wrapper(t, x):
            return self.dynamics(x, t)

        sol = solve_ivp(dynamics_wrapper, (0, t_eval[-1]), x0,
                        t_eval=t_eval, method='RK45')

        return sol.y.T[1:]  # Skip initial condition


class LyapunovExponent(BaseClass):
    """
    Lyapunov exponent computation for dynamical systems.

    Measures rate of separation of infinitesimally close trajectories.

    Args:
        dynamics: Function f(x, t) returning dx/dt
        dim: State space dimension
    """

    def __init__(self, dynamics: Callable[[np.ndarray, float], np.ndarray],
                 dim: int):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dim

    def jacobian(self, x: np.ndarray, t: float = 0,
                 epsilon: float = 1e-8) -> np.ndarray:
        """Compute Jacobian at (x, t)."""
        J = np.zeros((self.dim, self.dim))
        f0 = self.dynamics(x, t)

        for j in range(self.dim):
            x_pert = x.copy()
            x_pert[j] += epsilon
            J[:, j] = (self.dynamics(x_pert, t) - f0) / epsilon

        return J

    def largest_exponent(self, x0: ArrayLike, t_max: float,
                         dt: float = 0.01, n_renorm: int = 100) -> float:
        """
        Compute largest Lyapunov exponent using tangent vector evolution.

        Args:
            x0: Initial condition
            t_max: Total integration time
            dt: Time step
            n_renorm: Steps between renormalization

        Returns:
            Largest Lyapunov exponent
        """
        x0 = np.asarray(x0)
        x = x0.copy()

        # Initial tangent vector (perturbation direction)
        w = np.random.randn(self.dim)
        w = w / np.linalg.norm(w)

        lyap_sum = 0.0
        n_steps = int(t_max / dt)
        t = 0.0

        for i in range(n_steps):
            # Evolve state
            k1 = self.dynamics(x, t)
            k2 = self.dynamics(x + 0.5*dt*k1, t + 0.5*dt)
            k3 = self.dynamics(x + 0.5*dt*k2, t + 0.5*dt)
            k4 = self.dynamics(x + dt*k3, t + dt)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Evolve tangent vector
            J = self.jacobian(x, t)
            w = w + dt * (J @ w)

            # Renormalize periodically
            if (i + 1) % n_renorm == 0:
                norm = np.linalg.norm(w)
                lyap_sum += np.log(norm)
                w = w / norm

            t += dt

        return lyap_sum / t_max

    def spectrum(self, x0: ArrayLike, t_max: float,
                 dt: float = 0.01, n_renorm: int = 100) -> np.ndarray:
        """
        Compute full Lyapunov spectrum using QR decomposition.

        Args:
            x0: Initial condition
            t_max: Total integration time
            dt: Time step
            n_renorm: Steps between QR orthonormalization

        Returns:
            Array of Lyapunov exponents (sorted descending)
        """
        x0 = np.asarray(x0)
        x = x0.copy()

        # Initialize orthonormal frame
        Q = np.eye(self.dim)

        lyap_sums = np.zeros(self.dim)
        n_steps = int(t_max / dt)
        t = 0.0

        for i in range(n_steps):
            # Evolve state
            k1 = self.dynamics(x, t)
            k2 = self.dynamics(x + 0.5*dt*k1, t + 0.5*dt)
            k3 = self.dynamics(x + 0.5*dt*k2, t + 0.5*dt)
            k4 = self.dynamics(x + dt*k3, t + dt)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Evolve tangent vectors
            J = self.jacobian(x, t)
            Q = Q + dt * (J @ Q)

            # QR orthonormalization periodically
            if (i + 1) % n_renorm == 0:
                Q, R = np.linalg.qr(Q)
                lyap_sums += np.log(np.abs(np.diag(R)))

            t += dt

        return lyap_sums / t_max


class StrangeAttractor(BaseClass):
    """
    Strange attractor analysis and characterization.

    Args:
        trajectory: Trajectory on attractor [n_points, dim]
    """

    def __init__(self, trajectory: ArrayLike):
        super().__init__()
        self.trajectory = np.asarray(trajectory)
        self.n_points, self.dim = self.trajectory.shape

    @classmethod
    def from_dynamics(cls, dynamics: Callable, x0: ArrayLike,
                      t_transient: float, t_record: float,
                      dt: float = 0.01) -> 'StrangeAttractor':
        """
        Create attractor from dynamics by integrating.

        Args:
            dynamics: Function f(x, t) returning dx/dt
            x0: Initial condition
            t_transient: Time to discard (transient)
            t_record: Time to record
            dt: Time step
        """
        x0 = np.asarray(x0)

        def dynamics_wrapper(t, x):
            return dynamics(x, t)

        # Discard transient
        sol_trans = solve_ivp(dynamics_wrapper, (0, t_transient), x0, method='RK45')
        x_start = sol_trans.y[:, -1]

        # Record attractor
        t_eval = np.arange(0, t_record, dt)
        sol = solve_ivp(dynamics_wrapper, (0, t_record), x_start,
                        t_eval=t_eval, method='RK45')

        return cls(sol.y.T)

    @property
    def centroid(self) -> np.ndarray:
        """Attractor centroid (mean position)."""
        return np.mean(self.trajectory, axis=0)

    @property
    def extent(self) -> np.ndarray:
        """Extent in each dimension (max - min)."""
        return np.max(self.trajectory, axis=0) - np.min(self.trajectory, axis=0)

    def correlation_dimension(self, r_range: Optional[Tuple[float, float]] = None,
                              n_points_sample: int = 1000) -> float:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.

        D_2 = lim_{r→0} log(C(r)) / log(r)

        where C(r) is the correlation integral.
        """
        # Sample points if trajectory is large
        if self.n_points > n_points_sample:
            idx = np.random.choice(self.n_points, n_points_sample, replace=False)
            points = self.trajectory[idx]
        else:
            points = self.trajectory

        # Compute pairwise distances
        distances = pdist(points)

        if r_range is None:
            r_min = np.percentile(distances, 1)
            r_max = np.percentile(distances, 50)
            r_range = (r_min, r_max)

        # Compute correlation integral for various r
        r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), 20)
        C_values = []

        for r in r_values:
            C = np.mean(distances < r)
            C_values.append(max(C, 1e-10))

        # Linear fit in log-log space
        log_r = np.log(r_values)
        log_C = np.log(C_values)

        # Fit slope
        coeffs = np.polyfit(log_r, log_C, 1)

        return coeffs[0]  # Correlation dimension


class FractalDimension(BaseClass):
    """
    Fractal dimension estimation for attractors and sets.

    Implements multiple methods: box-counting, correlation, information.

    Args:
        points: Point set [n_points, dim]
    """

    def __init__(self, points: ArrayLike):
        super().__init__()
        self.points = np.asarray(points)
        self.n_points, self.dim = self.points.shape

    def box_counting(self, n_scales: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate box-counting dimension.

        D_0 = lim_{ε→0} log(N(ε)) / log(1/ε)

        Returns:
            dimension, box_sizes, box_counts
        """
        # Normalize points to [0, 1]^d
        points_norm = self.points - np.min(self.points, axis=0)
        points_norm = points_norm / np.max(points_norm)

        # Range of box sizes
        box_sizes = np.logspace(-3, 0, n_scales)
        box_counts = []

        for eps in box_sizes:
            # Count occupied boxes
            boxes = np.floor(points_norm / eps).astype(int)
            unique_boxes = np.unique(boxes, axis=0)
            box_counts.append(len(unique_boxes))

        box_counts = np.array(box_counts)

        # Linear fit
        log_eps_inv = np.log(1.0 / box_sizes)
        log_N = np.log(box_counts)

        # Fit only the scaling region
        coeffs = np.polyfit(log_eps_inv, log_N, 1)

        return coeffs[0], box_sizes, box_counts

    def information_dimension(self, n_scales: int = 20) -> float:
        """
        Estimate information dimension.

        D_1 = lim_{ε→0} Σ p_i log(p_i) / log(ε)
        """
        points_norm = self.points - np.min(self.points, axis=0)
        points_norm = points_norm / np.max(points_norm)

        box_sizes = np.logspace(-3, 0, n_scales)
        info_values = []

        for eps in box_sizes:
            boxes = np.floor(points_norm / eps).astype(int)
            # Count points in each box
            unique, counts = np.unique(boxes, axis=0, return_counts=True)
            probs = counts / self.n_points
            # Information entropy
            info = -np.sum(probs * np.log(probs))
            info_values.append(info)

        info_values = np.array(info_values)
        log_eps = np.log(box_sizes)

        coeffs = np.polyfit(log_eps, info_values, 1)

        return coeffs[0]


class RecurrencePlot(BaseClass):
    """
    Recurrence plot analysis for dynamical systems.

    R_ij = Θ(ε - ||x_i - x_j||)

    Args:
        trajectory: Time series trajectory [n_points, dim]
        epsilon: Recurrence threshold
    """

    def __init__(self, trajectory: ArrayLike, epsilon: float):
        super().__init__()
        self.trajectory = np.asarray(trajectory)
        self.epsilon = epsilon
        self.n_points = len(trajectory)

        self._compute_recurrence_matrix()

    def _compute_recurrence_matrix(self):
        """Compute recurrence matrix."""
        distances = squareform(pdist(self.trajectory))
        self.R = (distances < self.epsilon).astype(int)

    @property
    def recurrence_rate(self) -> float:
        """Recurrence rate: fraction of recurrence points."""
        return np.sum(self.R) / self.n_points**2

    @property
    def determinism(self) -> float:
        """
        Determinism: fraction of recurrence points forming diagonal lines.

        DET = Σ_{l≥l_min} l P(l) / Σ_{l≥1} l P(l)
        """
        # Find diagonal lines
        diagonal_lengths = []
        for offset in range(-self.n_points + 1, self.n_points):
            diag = np.diag(self.R, offset)
            # Count consecutive 1s
            current_length = 0
            for val in diag:
                if val == 1:
                    current_length += 1
                elif current_length > 0:
                    diagonal_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                diagonal_lengths.append(current_length)

        if not diagonal_lengths:
            return 0.0

        diagonal_lengths = np.array(diagonal_lengths)
        l_min = 2

        total = np.sum(diagonal_lengths)
        det_sum = np.sum(diagonal_lengths[diagonal_lengths >= l_min])

        return det_sum / total if total > 0 else 0.0

    @property
    def laminarity(self) -> float:
        """
        Laminarity: fraction of recurrence points in vertical lines.
        """
        vertical_lengths = []
        for j in range(self.n_points):
            col = self.R[:, j]
            current_length = 0
            for val in col:
                if val == 1:
                    current_length += 1
                elif current_length > 0:
                    vertical_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                vertical_lengths.append(current_length)

        if not vertical_lengths:
            return 0.0

        vertical_lengths = np.array(vertical_lengths)
        l_min = 2

        total = np.sum(vertical_lengths)
        lam_sum = np.sum(vertical_lengths[vertical_lengths >= l_min])

        return lam_sum / total if total > 0 else 0.0


class LorenzSystem(BaseClass):
    """
    Lorenz system - the canonical chaotic attractor.

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Args:
        sigma: Prandtl number (default 10)
        rho: Rayleigh number (default 28)
        beta: Geometric factor (default 8/3)
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0/3.0):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def dynamics(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """Lorenz equations."""
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ])

    @property
    def fixed_points(self) -> List[np.ndarray]:
        """Analytical fixed points."""
        points = [np.array([0, 0, 0])]

        if self.rho > 1:
            sqrt_term = np.sqrt(self.beta * (self.rho - 1))
            points.append(np.array([sqrt_term, sqrt_term, self.rho - 1]))
            points.append(np.array([-sqrt_term, -sqrt_term, self.rho - 1]))

        return points

    def integrate(self, x0: ArrayLike, t_span: Tuple[float, float],
                  n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Lorenz system."""
        x0 = np.asarray(x0)
        t = np.linspace(t_span[0], t_span[1], n_points)

        trajectory = odeint(lambda x, t: self.dynamics(x, t), x0, t)
        return t, trajectory

    def get_attractor(self, x0: ArrayLike = None,
                      t_transient: float = 50.0,
                      t_record: float = 100.0) -> StrangeAttractor:
        """Generate attractor trajectory."""
        if x0 is None:
            x0 = np.array([1.0, 1.0, 1.0])

        return StrangeAttractor.from_dynamics(
            self.dynamics, x0, t_transient, t_record)


class RosslerSystem(BaseClass):
    """
    Rössler system - simpler chaotic attractor.

    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)

    Args:
        a, b, c: System parameters
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def dynamics(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """Rössler equations."""
        x, y, z = state
        return np.array([
            -y - z,
            x + self.a * y,
            self.b + z * (x - self.c)
        ])

    def integrate(self, x0: ArrayLike, t_span: Tuple[float, float],
                  n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Rössler system."""
        x0 = np.asarray(x0)
        t = np.linspace(t_span[0], t_span[1], n_points)

        trajectory = odeint(lambda x, t: self.dynamics(x, t), x0, t)
        return t, trajectory

    def get_attractor(self, x0: ArrayLike = None,
                      t_transient: float = 100.0,
                      t_record: float = 200.0) -> StrangeAttractor:
        """Generate attractor trajectory."""
        if x0 is None:
            x0 = np.array([1.0, 1.0, 0.0])

        return StrangeAttractor.from_dynamics(
            self.dynamics, x0, t_transient, t_record)


class HenonMap(BaseClass):
    """
    Hénon map - canonical 2D discrete chaotic system.

    x_{n+1} = 1 - a*x_n² + y_n
    y_{n+1} = b*x_n

    Args:
        a: Nonlinearity parameter (default 1.4)
        b: Dissipation parameter (default 0.3)
    """

    def __init__(self, a: float = 1.4, b: float = 0.3):
        super().__init__()
        self.a = a
        self.b = b

    def map(self, state: np.ndarray) -> np.ndarray:
        """One iteration of Hénon map."""
        x, y = state
        return np.array([
            1 - self.a * x**2 + y,
            self.b * x
        ])

    def iterate(self, x0: ArrayLike, n_iterations: int) -> np.ndarray:
        """Iterate map n times."""
        trajectory = [np.asarray(x0)]
        state = trajectory[0].copy()

        for _ in range(n_iterations):
            state = self.map(state)
            trajectory.append(state.copy())

        return np.array(trajectory)

    @property
    def fixed_points(self) -> List[np.ndarray]:
        """Fixed points of Hénon map."""
        # x = 1 - ax² + bx => ax² + (1-b)x - 1 = 0
        # But actually: x = 1 - ax² + y, y = bx => x = 1 - ax² + bx
        # => ax² - (b-1)x - 1 = 0 => ax² + (1-b)x - 1 = 0
        discriminant = (1 - self.b)**2 + 4*self.a
        if discriminant < 0:
            return []

        sqrt_d = np.sqrt(discriminant)
        x1 = (-(1 - self.b) + sqrt_d) / (2*self.a)
        x2 = (-(1 - self.b) - sqrt_d) / (2*self.a)

        return [np.array([x1, self.b * x1]), np.array([x2, self.b * x2])]

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix at given state."""
        x, _ = state
        return np.array([
            [-2*self.a*x, 1],
            [self.b, 0]
        ])

    def lyapunov_exponent(self, x0: ArrayLike, n_iterations: int = 10000) -> float:
        """Compute largest Lyapunov exponent."""
        state = np.asarray(x0)
        lyap_sum = 0.0

        for _ in range(n_iterations):
            J = self.jacobian(state)
            # Use largest singular value
            lyap_sum += np.log(np.linalg.norm(J, ord=2))
            state = self.map(state)

        return lyap_sum / n_iterations


class LogisticMap(BaseClass):
    """
    Logistic map - simplest chaotic system.

    x_{n+1} = r*x_n*(1 - x_n)

    Shows period doubling route to chaos.

    Args:
        r: Growth rate parameter
    """

    def __init__(self, r: float = 3.9):
        super().__init__()
        self.r = r

    def map(self, x: float) -> float:
        """One iteration of logistic map."""
        return self.r * x * (1 - x)

    def iterate(self, x0: float, n_iterations: int) -> np.ndarray:
        """Iterate map n times."""
        trajectory = [x0]
        x = x0

        for _ in range(n_iterations):
            x = self.map(x)
            trajectory.append(x)

        return np.array(trajectory)

    @property
    def fixed_points(self) -> List[float]:
        """Fixed points: x* = 0 and x* = (r-1)/r."""
        points = [0.0]
        if self.r > 1:
            points.append((self.r - 1) / self.r)
        return points

    def stability(self, x_fixed: float) -> float:
        """Stability of fixed point: |f'(x*)| < 1 for stability."""
        # f'(x) = r(1 - 2x)
        return abs(self.r * (1 - 2*x_fixed))

    def bifurcation_diagram(self, r_range: Tuple[float, float] = (2.5, 4.0),
                            n_params: int = 1000,
                            n_iterations: int = 1000,
                            n_last: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bifurcation diagram."""
        r_values = np.linspace(r_range[0], r_range[1], n_params)
        all_r = []
        all_x = []

        for r in r_values:
            self.r = r
            x = 0.5

            # Transient
            for _ in range(n_iterations - n_last):
                x = self.map(x)

            # Record
            for _ in range(n_last):
                x = self.map(x)
                all_r.append(r)
                all_x.append(x)

        return np.array(all_r), np.array(all_x)

    def lyapunov_exponent(self, x0: float = 0.5,
                          n_iterations: int = 10000) -> float:
        """Compute Lyapunov exponent."""
        x = x0
        lyap_sum = 0.0

        for _ in range(n_iterations):
            # |f'(x)| = |r(1 - 2x)|
            derivative = abs(self.r * (1 - 2*x))
            if derivative > 0:
                lyap_sum += np.log(derivative)
            x = self.map(x)

        return lyap_sum / n_iterations


class DoublePendulumChaos(BaseClass):
    """
    Double pendulum - mechanical system exhibiting chaos.

    Uses Lagrangian mechanics with angles θ₁, θ₂ as generalized coordinates.

    Args:
        m1, m2: Masses (kg)
        L1, L2: Lengths (m)
        g: Gravitational acceleration (m/s²)
    """

    def __init__(self, m1: float = 1.0, m2: float = 1.0,
                 L1: float = 1.0, L2: float = 1.0, g: float = 9.81):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def dynamics(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Equations of motion for double pendulum.

        State: [θ₁, θ₂, ω₁, ω₂] where ω = dθ/dt
        """
        theta1, theta2, omega1, omega2 = state

        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        g = self.g

        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2 / L1) * den1

        dtheta1 = omega1
        dtheta2 = omega2

        domega1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * L2 * omega2**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / den1

        domega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / den2

        return np.array([dtheta1, dtheta2, domega1, domega2])

    def integrate(self, theta1_0: float, theta2_0: float,
                  omega1_0: float = 0.0, omega2_0: float = 0.0,
                  t_span: Tuple[float, float] = (0, 30),
                  n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate double pendulum motion."""
        x0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0])
        t = np.linspace(t_span[0], t_span[1], n_points)

        trajectory = odeint(lambda x, t: self.dynamics(x, t), x0, t)
        return t, trajectory

    def cartesian_positions(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert angular trajectory to Cartesian positions."""
        theta1 = trajectory[:, 0]
        theta2 = trajectory[:, 1]

        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)

        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)

        return np.column_stack([x1, y1]), np.column_stack([x2, y2])

    def total_energy(self, state: np.ndarray) -> float:
        """Total mechanical energy."""
        theta1, theta2, omega1, omega2 = state

        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        g = self.g

        # Kinetic energy
        T = 0.5 * m1 * (L1 * omega1)**2
        T += 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 +
                         2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))

        # Potential energy (reference at pivot)
        V = -m1 * g * L1 * np.cos(theta1)
        V -= m2 * g * (L1 * np.cos(theta1) + L2 * np.cos(theta2))

        return T + V


# Export all classes
__all__ = [
    'FixedPointAnalysis',
    'StabilityAnalysis',
    'PhasePortrait',
    'BifurcationDiagram',
    'PoincareSection',
    'LyapunovExponent',
    'StrangeAttractor',
    'FractalDimension',
    'RecurrencePlot',
    'LorenzSystem',
    'RosslerSystem',
    'HenonMap',
    'LogisticMap',
    'DoublePendulumChaos',
]
