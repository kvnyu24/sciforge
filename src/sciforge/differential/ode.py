import numpy as np

class ODESolver:
    """Base class for ODE solvers"""
    def __init__(self):
        pass
    
    def solve(self, f, y0, t_span, dt):
        raise NotImplementedError

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