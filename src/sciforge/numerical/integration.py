import numpy as np

def trapezoid(f, a, b, n):
    """Trapezoid rule for numerical integration"""
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    return h * (np.sum(f(x)) - 0.5*(f(a) + f(b)))

def simpson(f, a, b, n):
    """Simpson's rule for numerical integration"""
    if n % 2 == 0:
        n += 1
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    return h/3 * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 
                  2*np.sum(f(x[2:-1:2])) + f(x[-1])) 