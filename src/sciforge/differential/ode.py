from ..core.base import BaseSolver
from ..numerical.interpolation import cubic_spline
import numpy as np

class ODESolver(BaseSolver):
    """Base class for ODE solvers"""
    def __init__(self, store_history: bool = True):
        super().__init__(store_history)
    
    def validate_inputs(self, f, y0, t_span, dt):
        """Validate ODE solver inputs"""
        if not callable(f):
            raise TypeError("f must be callable")
        if not isinstance(y0, (np.ndarray, list)):
            raise TypeError("y0 must be array-like")
        if not len(t_span) == 2:
            raise ValueError("t_span must be (t0, tf)")
        if dt <= 0:
            raise ValueError("dt must be positive")

class Euler(ODESolver):
    """Simple Euler method"""
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using Euler method
        f: function defining the ODE dy/dt = f(t, y) 
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t)-1):
            y[i+1] = y[i] + dt * f(t[i], y[i])
            
        return t, y

class ImprovedEuler(ODESolver):
    """Improved Euler method (Heun's method)"""
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using improved Euler method
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition 
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t)-1):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + dt, y[i] + dt*k1)
            y[i+1] = y[i] + dt/2 * (k1 + k2)
            
        return t, y

class RungeKutta2(ODESolver):
    """2nd order Runge-Kutta method"""
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using RK2 method
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t)-1):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + dt, y[i] + dt*k1)
            y[i+1] = y[i] + (dt/2)*(k1 + k2)
            
        return t, y

class RungeKutta3(ODESolver):
    """3rd order Runge-Kutta method"""
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using RK3 method
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t)-1):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
            k3 = f(t[i] + dt, y[i] - dt*k1 + 2*dt*k2)
            y[i+1] = y[i] + dt/6 * (k1 + 4*k2 + k3)
            
        return t, y

class RungeKutta4(ODESolver):
    """4th order Runge-Kutta method"""
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using RK4 method
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t)-1):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
            k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
            k4 = f(t[i] + dt, y[i] + dt*k3)
            y[i+1] = y[i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return t, y

class AdamsBashforth(ODESolver):
    """Adams-Bashforth multi-step method"""
    def __init__(self, order=4):
        super().__init__()
        self.order = min(max(order, 1), 4)  # Limit order to 1-4
        
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using Adams-Bashforth method
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: time step
        """
        t0, tf = t_span
        t = np.arange(t0, tf + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        # Use RK4 to bootstrap first few points
        rk4 = RungeKutta4()
        _, y_start = rk4.solve(f, y0, (t0, t0 + (self.order-1)*dt), dt)
        y[:self.order] = y_start
        
        # Coefficients for different orders
        if self.order == 1:
            coef = [1]
        elif self.order == 2:
            coef = [3/2, -1/2]
        elif self.order == 3:
            coef = [23/12, -16/12, 5/12]
        else:  # order 4
            coef = [55/24, -59/24, 37/24, -9/24]
            
        # Main integration loop
        for i in range(self.order-1, len(t)-1):
            f_hist = [f(t[i-j], y[i-j]) for j in range(self.order)]
            y[i+1] = y[i] + dt * sum(c*f_val for c, f_val in zip(coef, f_hist))
            
        return t, y

class AdaptiveRK45(ODESolver):
    """Adaptive step size Runge-Kutta-Fehlberg method"""
    def __init__(self, rtol=1e-6, atol=1e-8):
        super().__init__()
        self.rtol = rtol
        self.atol = atol
        
    def solve(self, f, y0, t_span, dt):
        """
        Solve ODE using RK45 method with adaptive step size
        f: function defining the ODE dy/dt = f(t, y)
        y0: initial condition
        t_span: (t0, tf) time interval
        dt: initial time step
        """
        t0, tf = t_span
        t = [t0]
        y = [np.array(y0)]
        
        while t[-1] < tf:
            # RK45 coefficients
            k1 = dt * f(t[-1], y[-1])
            k2 = dt * f(t[-1] + dt/4, y[-1] + k1/4)
            k3 = dt * f(t[-1] + 3*dt/8, y[-1] + 3*k1/32 + 9*k2/32)
            k4 = dt * f(t[-1] + 12*dt/13, y[-1] + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = dt * f(t[-1] + dt, y[-1] + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            k6 = dt * f(t[-1] + dt/2, y[-1] - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            
            # 4th and 5th order solutions
            y4 = y[-1] + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            y5 = y[-1] + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
            
            # Error estimation
            error = np.max(np.abs(y5 - y4))
            tolerance = self.atol + self.rtol * np.max(np.abs(y[-1]))
            
            # Step size control
            if error <= tolerance:
                t.append(t[-1] + dt)
                y.append(y5)
                dt = 0.9 * dt * (tolerance/error)**(1/5)
            else:
                dt = 0.9 * dt * (tolerance/error)**(1/5)
            
            # Prevent overshooting tf
            if t[-1] + dt > tf:
                dt = tf - t[-1]
                
        return np.array(t), np.array(y)

class IVPSolver(ODESolver):
    """Initial Value Problem solver with adaptive step size control"""
    
    def __init__(self, method='RK45', rtol=1e-3, atol=1e-6, max_step=np.inf,
                 first_step=None, store_history=True):
        """
        Initialize IVP solver
        
        Args:
            method: Integration method ('RK45', 'RK23', 'DOP853', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum allowed step size
            first_step: Initial step size (estimated if None)
            store_history: Whether to store solution history
        """
        super().__init__(store_history)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.first_step = first_step
        
    def _estimate_first_step(self, fun, t0, y0, direction):
        """Estimate initial step size using RK error estimator"""
        if self.first_step is not None:
            return self.first_step
            
        scale = self.atol + np.abs(y0) * self.rtol
        f0 = fun(t0, y0)
        error_norm = np.linalg.norm(f0 / scale)
        
        if error_norm < 1e-10:
            h0 = 1e-6
        else:
            h0 = 0.01 * error_norm
            
        return h0 * direction
        
    def _rk_step(self, fun, t, y, h):
        """Perform single RK step with error estimate"""
        # Butcher tableau for RK45 (Cash-Karp)
        a = np.array([
            [0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0],
            [3/10, -9/10, 6/5, 0, 0],
            [-11/54, 5/2, -70/27, 35/27, 0],
            [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
        ])
        
        c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])
        b = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771])
        b_star = np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4])
        
        # Calculate RK stages
        k = np.zeros((6, len(y)))
        k[0] = fun(t, y)
        
        for i in range(1, 6):
            yi = y + h * sum(a[i,j] * k[j] for j in range(i))
            k[i] = fun(t + c[i] * h, yi)
            
        # Calculate solutions
        y_new = y + h * sum(b[i] * k[i] for i in range(6))
        y_error = h * sum((b[i] - b_star[i]) * k[i] for i in range(6))
        
        return y_new, y_error
        
    def solve(self, fun, t_span, y0, t_eval=None):
        """
        Solve initial value problem
        
        Args:
            fun: Right-hand side of dy/dt = f(t,y)
            t_span: Tuple of (t0, tf)
            y0: Initial state
            t_eval: Times at which to store the solution
            
        Returns:
            Solution object containing results
        """
        t0, tf = t_span
        direction = 1 if tf > t0 else -1
        
        if t_eval is None:
            t_eval = np.array([t0, tf])
        
        # Initialize solution storage
        t = [t0]
        y = [np.array(y0)]
        
        # Get initial step size
        h = self._estimate_first_step(fun, t0, y0, direction)
        
        # Main integration loop
        current_t = t0
        current_y = np.array(y0)
        
        while (direction * (tf - current_t) > 0):
            if direction * (current_t + h - tf) > 0:
                h = tf - current_t
                
            # Take step with error control
            y_new, error = self._rk_step(fun, current_t, current_y, h)
            
            # Error estimation and step size control
            scale = self.atol + np.maximum(np.abs(current_y), np.abs(y_new)) * self.rtol
            error_norm = np.linalg.norm(error / scale) / np.sqrt(len(y0))
            
            if error_norm < 1:
                # Step accepted
                current_t += h
                current_y = y_new
                
                # Store results
                t.append(current_t)
                y.append(current_y)
                
                # Update step size
                if error_norm < 0.1:
                    h *= min(10, 0.9 / error_norm ** 0.2)
                else:
                    h *= 0.9 / error_norm ** 0.2
                    
            else:
                # Reject step and reduce step size
                h *= max(0.1, 0.9 / error_norm ** 0.2)
                
            h = min(abs(h), self.max_step) * direction
            
        # Interpolate to get solution at requested times
        t = np.array(t)
        y = np.array(y)
        
        if self.store_history:
            self.save_state(t, y)
            
        return Solution(t, y, t_eval)

class Solution:
    """Container for ODE solution"""
    def __init__(self, t, y, t_eval):
        self.t = t
        self.y = y
        self.t_eval = t_eval
        self._interpolate()
        
    def _interpolate(self):
        """Interpolate solution to evaluation points"""
        
        # Interpolate each component
        y_components = []
        for i in range(self.y.shape[1]):
            spline = cubic_spline(self.t, self.y[:,i])
            y_components.append(spline(self.t_eval))
            
        self.y_eval = np.array(y_components).T