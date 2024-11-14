from ..core.base import BaseSolver
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